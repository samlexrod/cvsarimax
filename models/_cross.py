from sklearn.model_selection import TimeSeriesSplit
import types as ty
import numpy as np
import pandas as pd

class TimeSeriesComplexCV:
    def __init__(self, data_source, split_type='scikit', 
        validation_splits=2, enforece_complete_year=True, 
        enforece_test_homogenity=True,
        month_test_len=3):
        """
        
        """
        # attributes
        self.validation_splits = validation_splits
        self._data_source = data_source
        self._split_type = split_type
        self._force_complete = enforece_complete_year
        self._force_homogenity = enforece_test_homogenity
        self._is_list = isinstance(data_source, list)
        self._is_dataframe = isinstance(data_source, pd.DataFrame)
        self.month_test_len = month_test_len
        
        # error handling in data source
        if not self._is_dataframe and not self._is_list:
            raise ValueError("Must provide a generator or pandas dataframe as data_source") 
    
    def get_train_test_idx(self):
        """
        The user calls this method 
        """       
        # extracting attributes
        self._is_list = self._is_list
        data_source = self._data_source
        
        # decides if a loop is going to go over generator
        # or single dataframe
        if self._is_list:
            return self._apply_loop(data_source)
        else:
            return self._process_train_test_idx(data_source)
            
    def _apply_loop(self, data_list):
        """
        Used to apply a loop over a generator of dataframes
        """
        output_list = []
        for data in data_list:
            # ensure index is all date
            if not data.index.is_all_dates:
                data = data.set_index('date')
            gen_output = self._process_train_test_idx(data)
            output_list.append(gen_output)
        return output_list

    def _process_train_test_idx(self, df):
        # extracting 
        validation_splits = self.validation_splits
        split_type = self._split_type
        force_complete = self._force_complete
        force_homogenity = self._force_homogenity
        
        # error handling
        if validation_splits < 2:
            raise ValueError("The minimun split is 2 or number must be integer type")
        
        # extracting attributes
        split_type = split_type.lower()
        
        # adding service year and quarter
        months_dt = df.index.to_series().dt
        df = df.assign(
            year=months_dt.year,
            quarter=months_dt.quarter,
            month=months_dt.month
            )
        splits = validation_splits
        
        def quarter_cross_validation(df, force_complete):
            """
            """
            # adds the max quarters to find incomplete years            
            max_quarter_series = df\
                .groupby("year")\
                .quarter\
                .transform('max')
            df = df.assign(max_quarter=max_quarter_series)
            
            # adding last quarter includes january of next year
            # this is to indicate full year evaluations
            includes_january = df\
                .assign(month_shift=df.month.shift(-1))\
                .groupby("max_quarter").month_shift.transform('min') == 1
            df = df.assign(full_year_indicator=includes_january)
            print(df.reset_index())
            
            if force_complete:
                # ensure testing is done on complete years
                quarter_df = df[df.full_year_indicator == True]
            else:
                quarter_df = df

            # ignore last quarter if it is not complete
            full_last_quarter = df\
                .groupby(['year', 'quarter']).month.count()[-1] == 3
            if not full_last_quarter:
                df = df.iloc[:-1]

            # complete splits fo train and test list of list
            quarter_df = quarter_df.reset_index()
            index_max = df.reset_index().index.max()
            index_df = quarter_df.groupby(['year', 'quarter'])\
                .apply(lambda x: x.index.min())[-splits:]
            
            # generator loop
            for index_min in index_df.values:
                train_idx = np.arange(index_min)
                # ensure homogenity of testing where testing size
                # is the same accross validations
                if force_homogenity:
                    index_max = index_min + 2
                test_idx = np.arange(index_min, index_max+1)
                test_train_tuple = train_idx, test_idx
                yield test_train_tuple

        def month_cross_validation(df):            
            # reseting index and getting last lines
            index = df.reset_index().index
            index_max = index.max()
            # the split gets the last x indexes
            # these will be the start of the ranges
            start_idx = index[-splits:]
            test_len_adjust = self.month_test_len - 1
            
            # generator loop
            for index_min in start_idx:
                train_idx = np.arange(0, index_min-test_len_adjust)
                # ensure homogenity of testing where testing size
                # is the same accross validations
                if force_homogenity:
                    # avoids the test range 
                    # from going until the end
                    # and enforces all test
                    # to have same length
                    index_max = index_min

                # starting test right after train max range
                # ending right after the min index on 
                # force_homogenity or until the end index
                test_idx = np.arange(
                    index_min-test_len_adjust, 
                    index_max+1)
                test_train_tuple = train_idx, test_idx
                yield test_train_tuple
        
        def TimeSeriesSplit_validation(df):
            tscv = TimeSeriesSplit(n_splits=splits)
            return tscv.split(df) 
        
        if split_type in ('q', 'quarter', 'quarterly'):
            generator = quarter_cross_validation(df, force_complete)
        elif split_type in ('m', 'month', 'months', 'monthly'):
            generator = month_cross_validation(df)
        elif split_type in ('scikit', '', 'scikit-learn'):
            generator = TimeSeriesSplit_validation(df)
        else:
            raise ValueError("split_type not found")
        
        return generator
