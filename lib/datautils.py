import pandas as pd

def LoadData(datasetpath='ml-100k/u.data'):
    ratings = pd.read_csv(datasetpath, sep='\t', names=['user id','item id','rating','timestamp'], encoding = "UTF-8")
    users = pd.read_csv("Users.csv").drop("Unnamed: 0",axis=1)
    movies = pd.read_csv("Movies.csv").drop("Unnamed: 0",axis=1)
    ratings = ratings.rename(columns={'item id': 'movie id'})
    return users, movies, ratings


class DataProcessor:
    def __init__(self, users, movies, ratings):
        """
        Parameters:
        users (DataFrame): 用户数据, 包含user id和user features。
        movies (DataFrame): 电影数据, 包含movie id和movie features。
        ratings (DataFrame): 评分数据, 包含user id, movie id, rating和timestamp。
        """
        self.users = users
        self.movies = movies
        self.ratings = ratings

    def get_recent_movies(self, k=5):
        """
        获取每个用户最近看的k部电影。目前仅支持5.
        Returns:
        DataFrame: 格式为user id, movie1, ..., movie5, ...
        """
        # 首先对ratings数据按照时间戳进行排序
        sorted_ratings = self.ratings.sort_values(by='timestamp')
        # 使用groupby按照用户ID分组, 然后应用lambda函数取每组末尾的5条数据
        recent_movies = sorted_ratings.groupby('user id').tail(k)
        # 然后再调整格式, 以便每个用户ID仅占据一行
        recent_movies_df = recent_movies.groupby('user id')['movie id'].apply(list).reset_index()
        # 把列表分开成多个列
        recent_movies_df = pd.concat([recent_movies_df['user id'], recent_movies_df['movie id'].apply(pd.Series)], axis=1)
        recent_movies_df.columns = ['user id', 'movie1', 'movie2', 'movie3', 'movie4', 'movie5']
        return recent_movies_df

    def get_favorite_movies(self):
        """
        获取每个用户评分最高的5部电影。
        Returns:
        DataFrame: 格式为user id, movie1, ..., movie5以及相应的评分
        """
        sorted_ratings = self.ratings.sort_values(by=['user id', 'rating'], ascending=[True, False])
        top_movies = sorted_ratings.groupby('user id').head(5)
        top_movies_df = top_movies.groupby('user id').apply(lambda x: x[['movie id', 'rating']].values.tolist()).reset_index()
        # 展开嵌套列表到多个列
        top_movies_expanded = pd.DataFrame(
            top_movies_df[0].tolist(), index=top_movies_df.index)
        # 添加列名
        column_names = ['movie1_rat',  'movie2_rat', 'movie3_rat', 'movie4_rat',  'movie5_rat']
        top_movies_expanded.columns = column_names
        top_movies_df = pd.concat([top_movies_df['user id'], top_movies_expanded], axis=1)
        return top_movies_df

    def get_user_list(self):
        """
        Returns:
        List: 包含所有用户ID的列表。
        """
        return self.users['user id'].tolist()

    def get_movie_list(self):
        """
        Returns:
        List: 包含所有电影ID的列表。
        """
        return self.movies['movie id'].tolist()
    
    def count_user_movies_watched(self):
        """
        Returns:
        DataFrame: 一个包含用户ID和他们看过的电影数量的DataFrame。
        """
        watched_movies_count = self.ratings.groupby('user id')['movie id'].nunique().reset_index()
        watched_movies_count.columns = ['user id', 'movie_count']
        return watched_movies_count