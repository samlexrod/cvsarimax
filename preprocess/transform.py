import pandas as _pd

class PreProcess:
    def __init__(self, 
        dataframe, 
        grouping_column_list, 
        target_column, 
        index_column='date'):

        self.df = dataframe
        self.exog_list = self.df.drop([index_column, target_column] 
            + grouping_column_list, axis=1).columns.tolist()
        self.grouping_columns = grouping_column_list
        self.target_column = target_column
        self.index_column = index_column

        # error handling
        if target_column not in dataframe.columns:
            raise ValueError(
                "target_column not found in dataframe. "
                f"try from the following {dataframe.columns.to_list()}")
        if index_column not in dataframe.columns and index_column != dataframe.index.name:
            raise ValueError(
                "index_column not found in dataframe."
                f"try from the following {dataframe.columns.to_list()}")
        
        # creating preprocessing groupings
        grouping_list = dataframe.groupby(grouping_column_list).count().index.tolist() 
        self.grouping_list = grouping_list

        # indexing date column
        self._indexing_data()

    def _indexing_data(self):
        # extracting attributes
        df = self.df
        index_column = self.index_column

        # renames the date column and
        # indexes it if no date index
        # parsing date to datetime
        if not df.index.is_all_dates:
            self.df = df.rename(columns={index_column: 'date'})
            self.df['date'] = _pd.to_datetime(self.df.date)
            self._print_data_points()

            try:
                # updating df attribute
                self.df = self.df.set_index('date')                
            except:
                pass
            finally:
                # error handling
                # validating if date was indexed correctly
                if not self.df.index.is_all_dates:
                    raise ValueError(
                    f"Date indexed is not all in date format. "
                    "Pre-process the data to datetimeformat.")
                
                # setting the index frequency
                # self.df.index.freq = 'ms'
        
    def _print_data_points(self):
        # extracting attributes
        df = self.df
        grouping_columns = self.grouping_columns
        
        # rolling up on months
        try:
            data_point_df = df.assign(
                    year=lambda x: x.date.dt.year,
                    month=lambda x: x.date.dt.month)
            data_point_df = data_point_df\
                .groupby(grouping_columns + ['year']).month.nunique()\
                .groupby(grouping_columns).sum().to_frame("data_points")
            
            print("Data Points Available:",
                "\n" + "-"*50 + "\n",
                data_point_df,
                "\n" + "*"*50 + "\n")
        except Exception as e:
            raise ValueError("Grouping groups are incorrect...")
    
    def get_keys_and_dataframes(self, lags):
        # extracting attributes
        df = self.df
        target_columns = [self.target_column] + (self.exog_list or [])
        exog_columns = self.exog_list or []
        grouping_columns = self.grouping_columns
        grouping_list = self.grouping_list
        key_output_list = []
        data_output_list = []
        lag_exog_output_list = []
        
        # grouping data in df segments
        filter_format = ' and '.join(["%s == '%s'"] * len(grouping_columns))
        for group in grouping_list: 
            # creating the filter for the query method
            group = [group] if isinstance(group, str) else group
            column_group = list(zip(grouping_columns, group))
            column_group = [x for sub in column_group for x in sub]
            filter_query = filter_format % tuple(column_group)
            
            # creating the segment
            filtered_df = df.query(filter_query) 
            segment_count = filtered_df.shape[0]
            
            # pre-process if segment has data
            if segment_count > 0:
                
                # lag the segment by x months
                if lags > 0:
                    segment_df = filtered_df[:-lags]
                    segment_lag_df = filtered_df[-lags:]

                # creating the ouput lists
                key = (
                    segment_df[grouping_columns]
                        .groupby(grouping_columns)
                        .last().index.to_list()[0])
                key = key if isinstance(key, tuple) else [key]
                data = segment_df[target_columns]
                lag_data_exog = segment_lag_df[exog_columns]
                key_output_list.append(key)
                data_output_list.append(data)
                lag_exog_output_list.append(lag_data_exog)

        key_output_list = ['-'.join(k) for k in key_output_list]
        return key_output_list, data_output_list, lag_exog_output_list
