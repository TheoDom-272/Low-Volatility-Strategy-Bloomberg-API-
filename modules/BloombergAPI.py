import blpapi
import pandas as pd
import datetime as dt
import os


DATE = blpapi.Name("date")
ERROR_INFO = blpapi.Name("errorInfo")
EVENT_TIME = blpapi.Name("EVENT_TIME")
FIELD_DATA = blpapi.Name("fieldData")
FIELD_EXCEPTIONS = blpapi.Name("fieldExceptions")
FIELD_ID = blpapi.Name("fieldId")
SECURITY = blpapi.Name("security")
SECURITY_DATA = blpapi.Name("securityData")
MEMBER = blpapi.Name("Index Member")
WEIGHT =  blpapi.Name("Percent Weight")
INDX_WEIGHT = blpapi.Name("INDX_MWEIGHT_HIST")



#Class API Bloomberg
class BLP():
    # -----------------------------------------------------------------------------------------------------

    def __init__(self):
        """
            Improve this BLP object initialization Synchronus event handling
        """
        # Create Session object
        self.session = blpapi.Session()

        # Exit if can't start the Session
        if not self.session.start():
            print("Failed to start session.")
            return

        # Open & Get RefData Service or exit if impossible
        if not self.session.openService("//blp/refdata"):
            print("Failed to open //blp/refdata")
            return

        self.session.openService('//BLP/refdata')
        self.refDataSvc = self.session.getService('//BLP/refdata')

        print('Session open')

    # -----------------------------------------------------------------------------------------------------

    def bdh(self, strSecurity, strFields, startdate, enddate, per='DAILY', perAdj='CALENDAR',
            days='NON_TRADING_WEEKDAYS', fill='PREVIOUS_VALUE', currency=""):
        """
            Summary:
                HistoricalDataRequest ;

                Gets historical data for a set of securities and fields

            Inputs:
                strSecurity: list of str : list of tickers
                strFields: list of str : list of fields, must be static fields (e.g. px_last instead of last_price)
                startdate: date
                enddate
                per: periodicitySelection; daily, monthly, quarterly, semiannually or annually
                perAdj: periodicityAdjustment: ACTUAL, CALENDAR, FISCAL
                curr: string, else default currency is used
                Days: nonTradingDayFillOption : NON_TRADING_WEEKDAYS*, ALL_CALENDAR_DAYS or ACTIVE_DAYS_ONLY
                fill: nonTradingDayFillMethod :  PREVIOUS_VALUE, NIL_VALUE

                Options can be selected these are outlined in “Reference Services and Schemas Guide.”

            Output:
                A list containing as many dataframes as requested fields
            # Partial response : 6
            # Response : 5

        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('HistoricalDataRequest')

        # Put field and securities in list is single value is passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of securities
        for strF in strFields:
            request.append('fields', strF)

        for strS in strSecurity:
            request.append('securities', strS)

        # Set other parameters
        request.set('startDate', startdate.strftime('%Y%m%d'))
        request.set('endDate', enddate.strftime('%Y%m%d'))
        request.set('periodicitySelection', per)
        request.set('periodicityAdjustment', perAdj)
        request.set('nonTradingDayFillMethod', fill)
        request.set('nonTradingDayFillOption', days)
        if (currency != ""):
            request.set('currency', currency)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        dict_Security_Fields = {}
        liste_msg = []
        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            liste_msg.append(msg)
            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

        # -----------------------------------------------------------------------
        # Exploit data
        # ----------------------------------------------------------------------

        # Create dictionnary per field
        dict_output = {}
        for field in strFields:
            dict_output[field] = {}
            for ticker in strSecurity:
                dict_output[field][ticker] = {}

        # Loop on all messages
        for msg in liste_msg:
            countElement = 0
            security_data = msg.getElement(SECURITY_DATA)
            security = security_data.getElement(SECURITY).getValue()  # Ticker
            # Loop on dates
            for field_data in security_data.getElement(FIELD_DATA):

                # Loop on differents fields
                date = field_data.getElement(0).getValue()
                print(f"Traitement - Ticker: {security}, Date: {date.strftime('%Y-%m-%d')}")

                for i in range(1, field_data.numElements()):
                    field = field_data.getElement(i)
                    print(f"  Champ: {field.name()}, Valeur: {field.getValue()}")
                    dict_output[str(field.name())][security][date] = field.getValue()

                countElement = countElement + 1 if field_data.numElements() > 1 else countElement

            # remove ticker
            if countElement == 0:
                for field in strFields:
                    del dict_output[field][security]

        for field in dict_output:
            dict_output[field] = pd.DataFrame.from_dict(dict_output[field])
        return dict_output

        # -----------------------------------------------------------------------------------------------------

    def bdp(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        list_msg = []

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            list_msg.append(msg)
            #print(msg)

            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Extract the data
        # -----------------------------------------------------------------------

        dict_output = {}
        for msg in list_msg:

            for security_data in msg.getElement(SECURITY_DATA):
                ticker = security_data.getElement(SECURITY).getValue()  # Ticker
                dict_output[ticker] = {}
                for i in range(0, security_data.getElement(FIELD_DATA).numElements()):  # on boucle sur les fields
                    fieldData = security_data.getElement(FIELD_DATA).getElement(i)
                    dict_output[ticker][str(fieldData.name())] = fieldData.getValue()

        return pd.DataFrame.from_dict(dict_output).T




    def bds(self, strSecurity, strFields, strOverrideField='', strOverrideValue=''):

        """
            Summary:
                Reference Data Request ; Real-time if entitled, else delayed values
                Only supports 1 override


            Input:
                strSecurity
                strFields
                strOverrideField
                strOverrideValue

            Output:
               Dict
        """

        # -----------------------------------------------------------------------
        # Create request
        # -----------------------------------------------------------------------

        # Create request
        request = self.refDataSvc.createRequest('ReferenceDataRequest')

        # Put field and securities in list is single field passed
        if type(strFields) == str:
            strFields = [strFields]

        if type(strSecurity) == str:
            strSecurity = [strSecurity]

        # Append list of fields
        for strD in strFields:
            request.append('fields', strD)

        # Append list of securities
        for strS in strSecurity:
            request.append('securities', strS)

        # Add override
        if strOverrideField != '':
            o = request.getElement('overrides').appendElement()
            o.setElement('fieldId', strOverrideField)
            o.setElement('value', strOverrideValue)

        # -----------------------------------------------------------------------
        # Send request
        # -----------------------------------------------------------------------

        requestID = self.session.sendRequest(request)
        print("Sending request")

        # -----------------------------------------------------------------------
        # Receive request
        # -----------------------------------------------------------------------

        list_msg = []

        while True:
            event = self.session.nextEvent()

            # Ignores anything that's not partial or final
            if (event.eventType() != blpapi.event.Event.RESPONSE) & (
                    event.eventType() != blpapi.event.Event.PARTIAL_RESPONSE):
                continue

            # Extract the response message
            msg = blpapi.event.MessageIterator(event).__next__()
            # print(msg)

            # Break loop if response is final
            if event.eventType() == blpapi.event.Event.RESPONSE:
                break

                # -----------------------------------------------------------------------
        # Extract the data
        # -----------------------------------------------------------------------

        dict_output = {}

        for index_data in msg.getElement(SECURITY_DATA):
            index = index_data.getElement(SECURITY).getValue()  # Ticker
            dict_output[index] = {}
            for field_data in index_data.getElement(FIELD_DATA):
                for ele in field_data:
                    member = ele.getElement(MEMBER)
                    weight_member = ele.getElement(WEIGHT)

                    dict_output[index][member.getValue()] = weight_member.getValue()

        df_indices = {}
        for index, members_weights in dict_output.items():
            df_indices[index] = pd.DataFrame(list(members_weights.items()), columns=['Member', 'Weight'])

        return df_indices

    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------

    def closeSession(self):
        print("Session closed")
        self.session.stop()




