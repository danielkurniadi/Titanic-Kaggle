import numpy as np
import pandas as pd
import sklearn

###########################################################################################

def makeRegisterFunc():
    registered_func = {}
    def registfunc(func):
        registered_func[func.__name__] = func
        return func
    registfunc.all = registered_func
    return registfunc

###########################################################################################

class TitanicBasePreprocessing():
    def __init__(self, train_filepath, test_filepath):
        self.og_train_df = pd.read_csv(train_filepath)
        self.og_test_df = pd.read_csv(test_filepath)

        self.train_df = self.og_train_df
        self.test_df = self.og_test_df

    ######## Cleaning #########

    def __drop_features(self, features):
        pass

    def __drop_nan_features(self, df, features):
        """Clean missing values in each columns/features by dropping rows with Nan values in
        those columns
        """
        for feat in features:
            df = df[pd.notnull(df[feat])]

        return df
        
    def __fill_nan_byregression(self, df, target, reg_features):
        """Fill missing values in a column by using regression. Note that null/nan is interchangable here
        """
        #setup notnull and null data
        X_notnull, y_notnull = df[df[target].notnull()].loc[:, reg_features], df[df[target].notnull()][target]
        target_index = df[df[target].isnull()].index
        X_null = df[df[target].isnull()].loc[:, reg_features]
        #classifier
        r_forest_reg = RandomForestRegressor(n_estimators=400, n_jobs=4, max_depth=6)
        r_forest_reg.fit(X_notnull, y_notnull)
        #evaluate
        train_scores = cross_val_score(r_forest_reg, X_notnull, y_notnull,
                            scoring="neg_mean_squared_error", cv=5)
        print("Training accuracy score: ", train_scores.mean())
        #output prediction
        y_pred_nan = pd.Series(r_forest_reg.predict(X_null), index=age_index)

        return y_pred_nan
    
    ####### Create features ########

    def __create_fam_size(self, df):
        df['Fam_size'] = df['SibSp'] + df['Parch'] +1
        return df

    def __create_title_group(self, df):
        #inner helper function
        def __get_title(name):
            title_search = re.search(' ([A-Za-z]+)\. ', name)
            if title_search:
                return title_search.group(1)
            return ""

        df['Title'] = df['Name'].apply(__get_title)

        #groups of title
        rare_title = ['Master', 'Dr', 'Rev', 'Major', 'Col','Jonkheer', 'Sir', 'Capt', 'Don', 'Countess', 'Lady', 'Dona']
        miss_title = ['Miss', 'Mlle', 'Ms']
        madam_title = ['Mrs', 'Mme']
        man_title = ['Mr']

        #group title in rare and commons
        df['Title'] = df['Title'].replace(rare_title, 'RARE')
        df['Title'] = df['Title'].replace(miss_title, 'MISS')
        df['Title'] = df['Title'].replace(madam_title, 'MAM')
        df['Title'] = df['Title'].replace(man_title, 'MR')

        return df

    def __create_age_group(self, df, group_type='ordinal'):
        if group_type=='ordinal':
            labels = [1, 2, 3, 4]
        else:
            labels = ['CHILD', 'TEEN', 'ADULT', 'ELDER']
        df['Age_bin'] = pd.cut(df['Age'], bins=[0,13,25,50,120], labels=labels)
        
        return df

    def __create_new_features(self, df):
        pass