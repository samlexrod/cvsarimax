from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import adfuller
from sklearn.metrics import mean_squared_error
from datetime import datetime
import warnings
import numpy as np
import pandas as _pd
import types as ty
import secrets

class CvSARIMAX:
    def __init__(self, 
        pre_processed_data, 
        target_column,
        train_test_idx, 
        grouping_keys=None,
        cross_validate_params=None,
        exog_predictors=None,
        exog_lags=None):
        """
        Cross Validated SARIMAX model for time series forecasting.
        It process multiple sequential crossvalidations and generate evaluation 
        and forecasting data.
        
        parameters
        ----------
        pre_processed_data : the data to process in ARIMA. It has to be
            in pandas dataframe format and indexed by dates with an index 
            frequency (index.freq) like 'MS', etc.. The object
            can be of either a single dataframe, a list of dataframes, or
            a generator of dataframes.

            E.g.
            data = data.set_index('date')
            data.index.freq = 'MS'

        train_test_idx : the index splits of the dataframe to separate
            the training and the testing. The object can be of either a double
            list with the test and train indexes to apply to a dataframe; 
            a triple list with a list of test and train indexes for each dataframe;
            or a generator containing either a list or a generator of train and test
            to apply to each generated dataframe.

        Note: Make sure that the pre_processed_data argument matches the 
            training_idx_splits argument.

        cross_validate_params : These are the parameters to use in the cross_validate model.
            It must be in a dictionary format.

        returns
        -------
        It a dictionary of ...
        """

        # set attributes for internal use
        self.data_source = pre_processed_data
        self._target_column = target_column
        self._train_test_idx = train_test_idx
        self._grouping_keys = grouping_keys
        self._cross_validate_params = cross_validate_params
        self._exog_predictors = exog_predictors
        optimal_qualifier = cross_validate_params.get('optimal_qualifier', 'aic')
        if isinstance(optimal_qualifier, str): 
            optimal_qualifier = [optimal_qualifier]
        self._optimal_qualifier = optimal_qualifier
        self._exog_fit = exog_lags

        # generating run_time and run_key
        self._run_time = datetime.now()
        self._run_key = secrets.token_hex(16)  

        # print the current instantiation
        dataframe = pre_processed_data[0] if type(pre_processed_data) == list else pre_processed_data
        print(f"""
    MODEL INSTANTIATION SETTINGS:
        Dataframe Columns: {dataframe.columns}
        Target Column: {target_column}
        Contains Exogenous: {dataframe.shape[1] > 1}
        Exogenous Variables: {dataframe.columns.drop(target_column)}
        Using Exogenouse Lags: {True if exog_lags else False}
        Grouping Structure: {grouping_keys}
        Optimal Qualifier: {self._optimal_qualifier}
        """)

    def cross_validate(self, ignore_error=False):
        """
        The user uses this method to call __process_evaluation_loop()
        when data is provider in list of dataframes fromat. Otherwise,
        the model runs just once.
        """
        print(f"{'*'*65}\n*\t\t\tCROSS VALIDATION\t\t\t*\n{'*'*65}")
        
        # setting internal attributes
        self._ignore_error = ignore_error

        # extracting attributes
        data_source = self.data_source
        train_test_idx = self._train_test_idx

        # decide which running plan is going to be selected
        data_is_list = isinstance(data_source, list)
        data_is_dataframe = isinstance(data_source, _pd.DataFrame)
        if data_is_dataframe:
            idx_is_list = (isinstance(train_test_idx, list)) & (len(train_test_idx) == 2)

        if data_is_list:
            # runs the list of dataframes in the loop
            return self.__process_evaluation_loop()
        if data_is_dataframe and idx_is_list:
            # convert the dataframe into a list of 1
            data_source = [data_source]
            return self.__process_evaluation_loop()
        else:
            raise ValueError("Instantiation failed. Ensure consistency of source and train/test argumets")

    def fit_optimal(self):
        """
        The user uses this method to call __process_model_loop()
        when data is provider in list of dataframes fromat. Otherwise,
        the model runs just once.
        """
        print(f"{'*'*73}\n*\t\t\t\tFITTING MODEL\t\t\t\t*\n{'*'*73}")

        # extracting attributes
        data_source = self.data_source

        # decide which running plan is going to be selected
        data_is_list = isinstance(data_source, list)
        data_is_dataframe = isinstance(data_source, _pd.DataFrame)

        if data_is_list:
            return self.__process_model_loop()
        if data_is_dataframe:
            # convert the dataframe into a list of 1
            data_source = [data_source]
            return self.__process_model_loop()

    def __process_model_loop(self):
        """
        
        """
        print("*"*73)
        print("* \t\t\tPROCESSING OPTIMAL MODEL  \t\t\t*")
        print("*"*73)
        

        # starting with empty dataframe
        forecast_df = _pd.DataFrame()
        history_forecast_data = _pd.DataFrame()
        forecast_stats_dic = dict(
            run_key=[],
            run_time=[],
            group_key=[],
            hist_aic=[],
            hist_bic=[],
            hist_het=[],
            hist_jap=[],
            hist_ljp=[]
        )

        # extracting attributes
        data_source = self.data_source
        optimal_hyper_params = self._optimal_hyper_parameters_list
        grouped_exog_list = self._exog_fit        

        # zipping the hyper parameter and data list
        data_source_splits = zip(optimal_hyper_params, data_source, grouped_exog_list)

        # iterate over all data sources and hyper params to get forcast
        params = self._cross_validate_params
        horizon = params.get('horizon', 3)
        for optimal_param, data, exog in data_source_splits:
            # Determining if there is an exogenous variable
            contains_exog = False
            if data.shape[1] > 1:
                """It contains exog"""
                contains_exog = True
                target_column = self._target_column
                exog_columns = data.drop(target_column, axis=1).columns.tolist()

            key = optimal_param[-1]
            optimal_param = optimal_param[:-3]
            p, d, q, P, D, Q, s, t = optimal_param

            print("\n", "*"*100, f"\nProcessing: {key}")
            print(f"\tFitting SARIMAX({p}, {d}, {q})x({P}, {D}, {Q}, {s})"
                  f" on {t if t else 'no'} trend")

            if contains_exog:
                """Data contains exogenous variables"""
                optimal_model = SARIMAX(
                    data[target_column],
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    trend=t,
                    exog=data[exog_columns]
                )
            else:
                optimal_model = SARIMAX(
                    data,
                    order=(p, d, q),
                    seasonal_order=(P, D, Q, s),
                    trend=t
                )
            results = optimal_model.fit(disp=False)
            print(results.summary())

            # extract forecast results
            if contains_exog:
                exog_list = [list(x.values()) for x in exog.to_dict().values()]
                lags = len(exog_list[0])
                if horizon > lags: 
                    exog_list = [x + [0] * (horizon - lags) for x in exog_list]
                forecast_results = results.get_forecast(steps=horizon, exog=exog_list)
            else:
                forecast_results = results.get_forecast(steps=horizon)

            # extract prediction and intervals
            forecast_predictions = (
                forecast_results.predicted_mean.to_frame('prediction')
            )
            forecast_intervals = forecast_results.conf_int()

            # extract model statistics
            forecast_stats_dic['run_key'].append(self._run_key)            
            forecast_stats_dic['run_time'].append(self._run_time)
            forecast_stats_dic['group_key'].append(key)
            forecast_stats_dic['hist_aic'].append(results.aic)
            forecast_stats_dic['hist_bic'].append(results.bic)
            forecast_stats_dic['hist_het'].append(
                round(results.test_heteroskedasticity('breakvar')[0][1], 3)
            )
            forecast_stats_dic['hist_jap'].append(
                round(results.test_normality('jarquebera')[0][1], 3)
            )
            forecast_stats_dic['hist_ljp'].append(
                round(results.test_serial_correlation('ljungbox')[0][1][-1], 3)
            )

            def optimal_format(x):
                pdq = ''.join([str(y) for y in x[:3]])
                PDQ = ''.join([str(y) for y in x[3:-2]])
                S = str(x[-2])
                t = x[-1]
                return f"{pdq}-{PDQ}-{S}-{t}"

            # combine prediction with intervals
            forecast_intervals.columns = ['lower_ci', 'upper_ci']
            prediction_df = forecast_predictions.join(forecast_intervals)
            prediction_df.loc[:, 'run_key'] = self._run_key
            prediction_df.loc[:, 'group_key'] = key
            prediction_df.loc[:, 'optimal_parameters'] = optimal_format(optimal_param)

            # building full forecast dataframe
            forecast_df = _pd.concat([forecast_df, prediction_df])

            history_forecast_data = _pd.concat([history_forecast_data, 
                data.reset_index().merge(forecast_df.reset_index().rename(
                    columns={'index':'date'}), on='date', how='outer')])

        # converting stats dict to dataframe
        forecast_stats_df = _pd.DataFrame(forecast_stats_dic)

        # setting forecast outputs as attributes
        forecast_df.index.name = 'date'
        self.history_forecast_data = history_forecast_data
        self.forecast_df = forecast_df.reset_index()
        self.forecast_stats_df = forecast_stats_df

    def __process_evaluation_loop(self):
        """
        This internal process handles a generator
        or a list of dataframes
        """
        print("*"*73)
        print("* \t\t\tPROCESSING CROSS VALIDATIONS  \t\t\t*")
        print("*"*73)

        # starting with empty dataframe, slicing, and rank
        drop_pdqPDQs = ['p', 'd', 'q', 'P', 'D', 'Q', 's']
        # hyper_param_simple = ['pdqPDQs_key', 'trend',
        #     'group_optimal_rank', 'group_key', 'run_key', 'run_time']
        evaluation_df = _pd.DataFrame()
        evaluation_grouped = _pd.DataFrame()
        residual_trend_group = _pd.DataFrame()
        self._optimal_hyper_parameters_list = []

        # extract attributes
        keys = self._grouping_keys
        data_source = self.data_source
        train_test_idx = self._train_test_idx

        # zipping data_sources with training and testing indexes
        data_source_splits = zip(keys, data_source, train_test_idx)

        # iterate on data sources in generator or list of df
        for key, data, indexes in data_source_splits:
            # the inner loop is to handle the multiple splits
            # from cross validation
            print(f"Processing: \n", "-"*50)
            print(f"Group Key: {key}")
            print(f"Index: {data.index}")

            for i, split_idx in enumerate(indexes, 1):
                # if i == 1: continue
                print(f"{key} Cross Validation: {i}            ")
                print(f"\tTrain idx: {split_idx[0].min()} to {split_idx[0].max()}")
                print(f"\tTest idx: {split_idx[1].min()} to {split_idx[1].max()}")
                
                # **TESTING AND TRAINING SPLIT
                # extracting training and testing
                training_idx = split_idx[0]
                testing_idx = split_idx[1]
                train_data = data.iloc[training_idx]
                test_data = data.iloc[testing_idx]

                # declaring variable to extract column used as target
                # history_column = test_data.iloc[:, 0].name

                # extracting SARIMAX p, d, q at a train test individually
                d, s = self.__dicky_fuller_test(train_data)

                # processing SARIMAX on single training cross validation
                evaluation_individual, residual_trend_individual = self.__train_test_evaluations(
                    train_data, test_data, d, s)

                # adding columns to the ouput dataframe
                ## individual means at a cross validation level - no trend
                evaluation_individual.loc[:, 'run_key'] = self._run_key
                evaluation_individual.loc[:, 'run_time'] = self._run_time
                evaluation_individual.loc[:, 'group_key'] = key
                evaluation_individual.loc[:, 'cross_val_num'] = i
                evaluation_grouped = _pd.concat([evaluation_grouped, evaluation_individual], ignore_index=True)

                ## with trend
                residual_trend_individual.loc[:, 'run_key'] = self._run_key 
                residual_trend_individual.loc[:, 'group_key'] = key                
                residual_trend_individual.loc[:, 'cross_val_num'] = i
                residual_trend_group = _pd.concat([residual_trend_group, residual_trend_individual])
            
            # remove evaluations not meeting treshold at a key level            
            # resid_tresh = self._cross_validate_params.get('residual_treshold', .10)

            # the residual trend gets unioned at a key level
            ## it filters any iteration by cross validation that
            ## does not meet the treshold
            grouping = ['group_key', 'cross_val_num', 'pdqPDQs_key', 'trend', 'date']
            residual_trend_df = residual_trend_group.sort_values(grouping)
            residual_trend_df['abs_err_ratio'] = residual_trend_df.err_ratio.abs()            

                                          #
            ##########################    #
            # BUILDING THE THRESHOLD #  #####
            ##########################   ###
                                          #

            # smoothing the line on abnormal trends
            residual_trend_df['history_rolled'] = residual_trend_df['history'].rolling(12).median()
            abnormal_lambda = lambda x: x['covid'] == 1
            residual_trend_df.loc[abnormal_lambda, 'history_smoothed'] = residual_trend_df[abnormal_lambda].history_rolled
            residual_trend_df.history_smoothed.fillna(residual_trend_df['history'], inplace=True)
            residual_trend_df = residual_trend_df.assign(
                residual_smoothed=lambda x: x.history_smoothed - x.forecast.fillna(0),
                err_ratio_smoothed=lambda x: x.residual_smoothed / x.history_smoothed
            )
            residual_trend_df['abs_err_ratio_smoothed'] = residual_trend_df.err_ratio_smoothed.abs()

            # 1.
            # getting the max ratio per grouping
            trend_fillna = lambda x: x.trend.fillna('')
            residual_trend_df['max_err_ratio'] = residual_trend_df.assign(trend=trend_fillna)\
                .groupby(grouping[:-1]).abs_err_ratio_smoothed.transform('max')

            # 2.
            # setting the threshold tiers
            tier_lambda = lambda x: 1 if x <=.10 else 2 if x <=.20 else 3 if x <=.30 else 4 
            residual_trend_df['thresh_tier'] = residual_trend_df.max_err_ratio.apply(tier_lambda)

            # 3.
            # establishing the optimal indicators to filter optimal hyper params
            minimun_tier = residual_trend_df.groupby(['group_key', 'cross_val_num']).thresh_tier.transform('min')
            find_optimal_condition = residual_trend_df.thresh_tier == minimun_tier
            residual_trend_df['optimal_trend_indicator'] = find_optimal_condition

            # saving data to disk to do exploration of results in jupyter notebook
            # residual_trend_df.to_csv(r"H:\data\nrm-data\residual_trend.csv")

            # 4.
            # assigning the threshold as True and False in the evaluation
            optimal_key = ['pdqPDQs_key', 'trend', 'group_key', 'cross_val_num']
            optimal_df = residual_trend_df[lambda x: x.optimal_trend_indicator == True][optimal_key]
            optimal_df = optimal_df.groupby(optimal_key).count()
            optimal_df['optimal_trend_indicator'] = True
            evaluation_grouped = evaluation_grouped.join(optimal_df, on=optimal_key, how='left')
            # saving data to disk to do exploration of results in jupyter notebook
            # evaluation_grouped.to_csv(r'H:\data\nrm-data\evaluation_grouped.csv', index=False, header=True)            

            # 5.
            # assigning optimal rank to evaluation groups at the group_key level
            optimal_focus = self._optimal_qualifier
            evaluation_grouped['group_optimal_rank'] = (
                evaluation_grouped
                .where(lambda x: x.optimal_trend_indicator==True)
                .groupby(['group_key'])[optimal_focus] 
                .rank('first'))

            # 6.
            # assigning optimal rank to evaluation groups at the group_key level and cv 
            evaluation_grouped['group_cv_optimal_rank'] = (
                evaluation_grouped
                .where(lambda x: x.optimal_trend_indicator==True)
                .groupby(['group_key', 'cross_val_num'])[optimal_focus]
                .rank('first'))

            # setting evaluation optimal by cross validation
            evaluation_df = _pd.concat([evaluation_df, evaluation_grouped.drop(drop_pdqPDQs, axis=1)])
            # saving data to disk to do exploration of results in jupyter notebook
            # evaluation_df.to_csv(r'H:\data\nrm-data\evaluation_df.csv', index=False, header=True)

            #################################
            # END OF BUILDING THE THRESHOLD #
            #################################

            # setting the optimal parameter list for the fit_optimal method
            hyper_param_cols = ['p', 'd', 'q', 'P', 'D', 'Q', 's', 'trend', 
                'group_optimal_rank', 'group_cv_optimal_rank', 'group_key'
            ] + optimal_focus

            self._optimal_hyper_parameters_list.append(
                evaluation_grouped[hyper_param_cols]
                .sort_values(optimal_focus)
                .query("group_cv_optimal_rank==1")
                .drop(columns=optimal_focus)
                .values
                .tolist()[0]
            )   

            # resets the sets for new groupings
            evaluation_grouped = _pd.DataFrame()      

        # setting external attributes
        print("\nOPTIMAL HYPER PARAMETER LIST")
        print(self._optimal_hyper_parameters_list)

        # re-identifying optimal residual trends of cv
        optimal_selected_df = evaluation_df[lambda x: x.group_cv_optimal_rank == 1][optimal_key]
        optimal_selected_df['optimal_selected'] = True

        # dataframe as attributes
        self.residual_trend_df = (
            residual_trend_df.reset_index()
        ).merge(optimal_selected_df, on=optimal_key, how='left')
        self.residual_trend_df['optimal_selected'] = self.residual_trend_df.optimal_selected.fillna(False)


        self.evaluation_df = (
            evaluation_df
            .drop("run_time", axis=1)
        )        

    def __dicky_fuller_test(self, train_data, alpha=.05):
        """
        The augmented dicky fuller statistical test for correlation.
        It is used to find sationarity.
        parameters
        ----------
        train_data : training split of time series data indexed by date
        alpha : cutoff value to compare to the duckey fuller p-value
            alpha < p-value = evidence to reject the null hypothesis
        returns : a constant variable for parameter d of SARIMAX
        """
        def apply_test(data):
            target_column = data.columns.tolist()[0]
            adf_results = adfuller(data[target_column])
            return adf_results[1]

        # set trend differencing variables
        data_diff = train_data # diff at 0
        try:
            trend_p_value = apply_test(data_diff)
        except:
            # one to avoid the while loops
            trend_p_value = 1
        d = 0

        # continue to difference data on trend
        while alpha < trend_p_value:
            data_diff = data_diff.diff().dropna()
            d =+ 1
            try:
                trend_p_value = apply_test(data_diff)
            except:
                d = 0
                trend_p_value = None
                print("\tWARNING! Unable to infer difference value due to "
                    "lack of data points")
                break

        print(
            f"\tDicky-fuller Train Data Length: {train_data.shape[0]}\n"
            f"\tDicky-fuller d: {d}\n"
            f"\tDicky-fuller p-value: {trend_p_value}\n"
        )

        # set seasonality differencing variables
        s = 1
        try:
            season_p_value = apply_test(data_diff.diff(s).dropna())
        except:
            # one to avoid the while loops
            season_p_value = 1        

        # continue to difference on seasonality
        while alpha < season_p_value:
            data_diff_season = data_diff.diff(s).dropna()
            s += 1
            try:
                season_p_value = apply_test(data_diff_season)
            except:
                s = 0
                season_p_value = None
                print("\tWARNING! Unable to infer seasonality due to "
                    "lack of data points")
                break
        print(
            f"\tDicky-fuller Seasonal Length: {data_diff.shape[0]}\n"
            f"\tDicky-fuller s: {s}\n"
            f"\tDicky-fuller p-value: {season_p_value}\n"
        )

        return d, s
        
    def __train_test_evaluations(self, 
        train_data, test_data, d, s,
        horizon=3,
        arma_ordermax=(1, 1), 
        include_trend=False,
        include_seasonality=False,
        seasonality_cycle='month'):
        """
        Testing on SARIMAX using the dicky fuller test d results.
        Iterating over the maxiter on p and q holding d constant.
        parameters
        ----------
        data : the training set
        d : the dicky fuller test differentiation results
        s : the dicky fuller test for seasonality differentiation results
        maxiter : the number of iterations to test p and q
            squared so 10**2 = 100 tests of aic or bic
        returns the best model order (p, d, q) in a tuple
        """


        # Determining if there is an exogenous variable
        contains_exog = False
        if train_data.shape[1] > 1:
            """It contains exog"""
            contains_exog = True
            target_column = self._target_column
            exog_columns = train_data.drop(target_column, axis=1).columns.tolist()

            if self._exog_predictors:
                print("Exog Status: Using argument exogenous predictors")
            else:
                print("Exog Stauts: Using testing exogenous predictors")

        debug_counter = 0
        err = ''

        # extracting variables
        history_column = self._target_column

        # starting empty dataframes
        cross_validation_residual_df = _pd.DataFrame()

        # extracting provided parameters
        params = self._cross_validate_params
        arma_ordermax = params.get('arma_ordermax', arma_ordermax)
        optimal_qualifier = self._optimal_qualifier
        include_trend = params.get('include_trend', include_trend)
        include_seasonality = params.get('include_seasonality', include_seasonality)
        seasonality_cycle = params.get('seasonality_cycle', seasonality_cycle)

        result_dic = dict(
            p=[], d=[], q=[], P=[], D=[], Q=[], s=[], pdqPDQs_key=[],
            aic=[], bic=[], trend=[], 
            ts_err_ratio_mean=[], ts_err_ratio_std=[], ts_err_ratio_absmax=[], 
            tr_resid_mae=[], tr_resid_rmse=[],
            tr_heteroskeda_p=[], tr_ljungbox_p=[], 
            tr_jarquebera_p=[], ts_resid_mse=[], ts_resid_rsme=[])

        # conducting training and testing evaluations
        trends = [None, 'c', 'n', 't', 'ct']
        range_p = range(arma_ordermax[0])
        range_q = range(arma_ordermax[1])
        range_P = range_p if include_seasonality else [0]
        range_Q = range_q if include_seasonality else [0]
        range_D = range(0, 2) if include_seasonality else [0]
        step = (
            1 if seasonality_cycle == 'month'
            else 3 if seasonality_cycle == 'quarter' 
            else 12
        )
        range_s = (
            np.arange(2, 13, step=step)
            if include_seasonality 
            else [0]
        )
        if not include_trend: trends = [None]

        for P in range_P:
            for D in range_D:
                for Q in range_Q:
                    for s in range_s:
                        for p in range_p:
                            for q in range_q:
                                for t in trends:
                                    # DEBUG SECTION ********
                                    debug_counter += 1
                                    # if p + q == 0: continue
                                    # if s < 12: continue
                                    # ***********************

                                    try:
                                        # fitting model and set coefficients                                        
                                        pdqPDQs_key = f"{p}{d}{q}-{P}{D}{Q}-{s}"
                                        print(f"Fitting: {pdqPDQs_key} with {t}", flush=True, sep=' ', end='      \r')
                                        warnings.filterwarnings('ignore')
                                        if contains_exog:
                                            model = SARIMAX(
                                                train_data[target_column], 
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s), 
                                                trend=t,
                                                exog=train_data[exog_columns])
                                        else:
                                            model = SARIMAX(
                                                train_data, 
                                                order=(p, d, q),
                                                seasonal_order=(P, D, Q, s), 
                                                trend=t)

                                        err = 'Warning! Model error on %s with %s' % (pdqPDQs_key, t)
                                        results = model.fit(disp=False)
                                        err = 'Error! transfromation error on %s with %s' % (pdqPDQs_key, t)

                                        # extracting predictions and ci
                                        test_len = test_data.shape[0]
                                        if contains_exog:
                                            if self._exog_predictors:
                                                endog_predictors = self._exog_predictors
                                            else:
                                                endog_predictors = test_data[exog_columns]
                                            forecast_results = results.get_forecast(steps=test_len, exog=endog_predictors)
                                        else:
                                            forecast_results = results.get_forecast(steps=test_len)

                                        prediction = forecast_results.predicted_mean.to_frame('forecast')

                                        # extracting train evaluation metrics
                                        resid = results.resid # residuals across the training set
                                        train_mae = np.mean(np.abs(resid))
                                        train_rmse = np.sqrt(train_mae)
                                        t_heteroskeda_p = results.test_heteroskedasticity('breakvar')[0][1]
                                        t_ljungbox_p = 0 #results.test_serial_correlation('ljungbox')[0][1][-1]
                                        t_jarquebera_p = results.test_normality('jarquebera')[0][1]

                                        # # enforce prediction length same as test length
                                        # test_len = test_data.shape[0]
                                        # prediction = prediction.iloc[:test_len]

                                        # creating the test vs prediction dataframe
                                        # here starts the evaluation trend of residuals **
                                        test_data.rename(columns={history_column: 'history'}, inplace=True)
                                        cv_individual_residual_df = test_data.join(prediction)
                                        cv_individual_residual_df.loc[:, 'pdqPDQs_key'] = pdqPDQs_key
                                        cv_individual_residual_df.loc[:, 'trend'] = t

                                        # calculating residuals and percentage of residuals
                                        cv_individual_residual_df = cv_individual_residual_df.assign(
                                            residual=lambda x: x.history - prediction.forecast.fillna(0),
                                            err_ratio=lambda x: x.residual / x.history
                                        )

                                        # extracting descriptive statistics and inference
                                        err_ratio_mean = cv_individual_residual_df.err_ratio.mean()
                                        err_ratio_std = cv_individual_residual_df.err_ratio.std()
                                        ts_err_ratio_absmax = cv_individual_residual_df.err_ratio.abs().max()
                                        test_mse = mean_squared_error(test_data['history'], prediction.fillna(0))
                                        test_rmse = np.sqrt(test_mse)

                                        # appending all cross validation prediction trends
                                        cross_validation_residual_df = _pd.concat(
                                            [cross_validation_residual_df, cv_individual_residual_df]
                                        )  

                                        # populating dictionary of evaluations
                                        result_dic['p'].append(p)
                                        result_dic['d'].append(d)
                                        result_dic['q'].append(q)                                    
                                        result_dic['P'].append(P)
                                        result_dic['D'].append(D)
                                        result_dic['Q'].append(Q)                                    
                                        result_dic['s'].append(s)
                                        result_dic['pdqPDQs_key'].append(pdqPDQs_key)
                                        result_dic['aic'].append(results.aic)
                                        result_dic['bic'].append(results.bic)
                                        result_dic['trend'].append(t)
                                        result_dic['tr_resid_mae'].append(round(train_mae, 2))  
                                        result_dic['tr_resid_rmse'].append(round(train_rmse, 2))
                                        result_dic['tr_heteroskeda_p'].append(round(t_heteroskeda_p, 3))
                                        result_dic['tr_ljungbox_p'].append(round(t_ljungbox_p, 3))
                                        result_dic['tr_jarquebera_p'].append(round(t_jarquebera_p, 3))
                                        result_dic['ts_err_ratio_mean'].append(err_ratio_mean)
                                        result_dic['ts_err_ratio_std'].append(err_ratio_std)
                                        result_dic['ts_err_ratio_absmax'].append(ts_err_ratio_absmax)
                                        result_dic['ts_resid_mse'].append(round(test_mse, 2))
                                        result_dic['ts_resid_rsme'].append(round(test_rmse, 2))
                                        
                                    except Exception as e:
                                        if self._ignore_error == False:
                                            print(err, e)
        
        result_df = _pd.DataFrame(result_dic)
        return result_df.sort_values(optimal_qualifier), cross_validation_residual_df
