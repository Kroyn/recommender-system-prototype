import pandas as pd
import numpy as np
from datetime import datetime
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')

class MovieRecommendationPivotAnalysis:
    def __init__(self):
        self.movies_df = None
        self.ratings_df = None
        self.users_df = None
        os.makedirs('media', exist_ok=True)
    
    def load_data(self):
        self.movies_df = pd.read_csv('movies.csv', encoding='utf-8', on_bad_lines='skip')
        
        self.ratings_df = pd.read_csv('ratings.csv', encoding='utf-8', on_bad_lines='skip')
        self.ratings_df['date'] = pd.to_datetime(self.ratings_df['date'])
        self.ratings_df['rating_year'] = self.ratings_df['date'].dt.year
        self.ratings_df['rating_month'] = self.ratings_df['date'].dt.month
        self.ratings_df['rating_quarter'] = self.ratings_df['date'].dt.quarter
        self.ratings_df['date_only'] = self.ratings_df['date'].dt.strftime('%Y-%m-%d')
        
        self.users_df = pd.read_csv('users.csv', encoding='utf-8', on_bad_lines='skip')
        
        self.movies_df['main_genre'] = self.movies_df['genres'].str.split('|').str[0]
        
        self.users_df['age_group'] = pd.cut(self.users_df['age'], 
                                          bins=[17, 25, 35, 45, 55, 65],
                                          labels=['18-25', '26-35', '36-45', '46-55', '56-65'])
    
    def create_pivot_tables(self):
        merged_data = self.ratings_df.merge(self.movies_df, on='movieId').merge(self.users_df, on='userId')
        
        with pd.ExcelWriter('movie_recommendation_pivot_analysis.xlsx', engine='openpyxl') as writer:
            
            merged_data_export = merged_data.copy()
            merged_data_export['date'] = merged_data_export['date'].dt.strftime('%Y-%m-%d')
            merged_data_export.to_excel(writer, sheet_name='Основні дані', index=False)
            
            pivot_ratings = pd.pivot_table(
                merged_data,
                values=['rating', 'userId'],
                index=['main_genre'],
                columns=['rating_year'],
                aggfunc={'rating': 'mean', 'userId': 'count'},
                fill_value=0,
                observed=False
            )
            pivot_ratings.to_excel(writer, sheet_name='Аналіз рейтингів')
            
            pivot_demographic = pd.pivot_table(
                merged_data,
                values=['rating', 'userId'],
                index=['age_group'],
                columns=['gender', 'occupation'],
                aggfunc={'rating': 'mean', 'userId': 'count'},
                fill_value=0,
                observed=False
            )
            pivot_demographic.to_excel(writer, sheet_name='Демографічний аналіз')
            
            genre_popularity = pd.pivot_table(
                merged_data,
                values=['rating', 'userId'],
                index=['main_genre'],
                aggfunc={'rating': 'mean', 'userId': 'count'},
                observed=False
            )
            genre_popularity.columns = ['Середній рейтинг', 'Кількість оцінок']
            genre_popularity = genre_popularity.sort_values('Кількість оцінок', ascending=False)
            genre_popularity.to_excel(writer, sheet_name='Популярність жанрів')
            
            time_trends = pd.pivot_table(
                merged_data,
                values=['rating', 'userId'],
                index=['rating_month'],
                columns=['rating_year'],
                aggfunc={'rating': 'mean', 'userId': 'count'},
                fill_value=0,
                observed=False
            )
            time_trends.to_excel(writer, sheet_name='Часові тренди')
            
            user_analysis = pd.pivot_table(
                merged_data,
                values=['rating', 'movieId'],
                index=['occupation'],
                columns=['gender', 'age_group'],
                aggfunc={'rating': 'mean', 'movieId': 'count'},
                fill_value=0,
                observed=False
            )
            user_analysis.to_excel(writer, sheet_name='Аналіз користувачів')
            
            top_movies = merged_data.groupby(['title', 'main_genre']).agg({
                'rating': ['mean', 'count'],
                'userId': 'nunique'
            }).round(2)
            top_movies.columns = ['Середній рейтинг', 'Кількість оцінок', 'Унікальні користувачі']
            top_movies = top_movies.sort_values('Кількість оцінок', ascending=False).head(50)
            top_movies.to_excel(writer, sheet_name='Топ фільмів')
            
            quarterly_analysis = pd.pivot_table(
                merged_data,
                values=['rating', 'userId'],
                index=['main_genre'],
                columns=['rating_quarter'],
                aggfunc={'rating': 'mean', 'userId': 'count'},
                fill_value=0,
                observed=False
            )
            quarterly_analysis.to_excel(writer, sheet_name='Квартальний аналіз')
            
            yearly_analysis = merged_data.groupby('rating_year').agg({
                'rating': 'mean',
                'userId': 'nunique',
                'movieId': 'nunique',
                'title': 'count'
            }).round(2)
            yearly_analysis.columns = ['Середній рейтинг', 'Унікальні користувачі', 'Унікальні фільми', 'Загальна кількість оцінок']
            yearly_analysis.to_excel(writer, sheet_name='Річний аналіз')
    
    def create_matlab_style_plots(self):
        merged_data = self.ratings_df.merge(self.movies_df, on='movieId').merge(self.users_df, on='userId')
        
        plt.style.use('default')
        
        fig = plt.figure(figsize=(18, 12))
        
        ax1 = fig.add_subplot(231, projection='3d')
        genre_stats = merged_data.groupby('main_genre').agg({
            'rating': 'mean',
            'userId': 'count',
            'movieId': 'nunique'
        }).reset_index()
        
        x = range(len(genre_stats))
        y = genre_stats['rating']
        z = genre_stats['userId']
        
        colors = plt.cm.viridis(np.linspace(0, 1, len(genre_stats)))
        scatter = ax1.scatter(x, y, z, c=colors, s=genre_stats['movieId']*5, alpha=0.7)
        ax1.set_xlabel('Жанри')
        ax1.set_ylabel('Середній рейтинг')
        ax1.set_zlabel('Кількість оцінок')
        ax1.set_title('3D Аналіз жанрів')
        
        ax2 = fig.add_subplot(232)
        yearly_activity = merged_data.groupby('rating_year').agg({
            'userId': 'count',
            'rating': 'mean'
        }).reset_index()
        
        ax2.plot(yearly_activity['rating_year'], yearly_activity['userId'], 
                marker='o', linewidth=2, markersize=4, color='blue', label='Кількість оцінок')
        ax2.set_xlabel('Рік')
        ax2.set_ylabel('Кількість оцінок', color='blue')
        ax2.tick_params(axis='y', labelcolor='blue')
        ax2.grid(True, alpha=0.3)
        
        ax2_2 = ax2.twinx()
        ax2_2.plot(yearly_activity['rating_year'], yearly_activity['rating'], 
                  marker='s', linewidth=1, markersize=3, color='red', label='Середній рейтинг')
        ax2_2.set_ylabel('Середній рейтинг', color='red')
        ax2_2.tick_params(axis='y', labelcolor='red')
        
        ax2.set_title('Активність по роках')
        
        ax3 = fig.add_subplot(233)
        age_gender_stats = pd.pivot_table(
            merged_data,
            values='rating',
            index='age_group',
            columns='gender',
            aggfunc='mean',
            observed=False
        )
        age_gender_stats.plot(kind='bar', ax=ax3, color=['lightblue', 'lightpink'])
        ax3.set_xlabel('Вікова група')
        ax3.set_ylabel('Середній рейтинг')
        ax3.set_title('Рейтинг за віком та статтю')
        ax3.legend(title='Стать')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(234)
        occupation_stats = merged_data.groupby('occupation').agg({
            'rating': 'mean',
            'userId': 'count'
        }).nlargest(8, 'userId')
        
        bars = ax4.bar(occupation_stats.index, occupation_stats['userId'], 
                      color=plt.cm.Set3(np.linspace(0, 1, len(occupation_stats))))
        ax4.set_xlabel('Професія')
        ax4.set_ylabel('Кількість оцінок')
        ax4.set_title('Топ професій за активністю')
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        ax5 = fig.add_subplot(235)
        rating_distribution = merged_data['rating'].value_counts().sort_index()
        ax5.pie(rating_distribution.values, labels=rating_distribution.index, 
               autopct='%1.1f%%', startangle=90, colors=plt.cm.Pastel1(range(len(rating_distribution))))
        ax5.set_title('Розподіл рейтингів')
        
        ax6 = fig.add_subplot(236)
        top_genres = merged_data['main_genre'].value_counts().head(6)
        ax6.bar(top_genres.index, top_genres.values, color=plt.cm.Set2(range(len(top_genres))))
        ax6.set_xlabel('Жанр')
        ax6.set_ylabel('Кількість оцінок')
        ax6.set_title('Топ жанрів')
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.subplots_adjust(left=0.05, right=0.95, top=0.92, bottom=0.08, wspace=0.3, hspace=0.4)
        plt.savefig('media/matlab_style_analysis.png', dpi=200, bbox_inches='tight')
        plt.show()
        
        self.create_additional_3d_plots(merged_data)
    
    def create_additional_3d_plots(self, merged_data):
        fig = plt.figure(figsize=(16, 10))
        
        ax1 = fig.add_subplot(221, projection='3d')
        user_activity = merged_data.groupby('userId').agg({
            'rating': ['mean', 'count'],
            'age': 'first'
        }).reset_index()
        user_activity.columns = ['userId', 'avg_rating', 'rating_count', 'age']
        
        x = user_activity['age']
        y = user_activity['avg_rating']
        z = user_activity['rating_count']
        
        scatter = ax1.scatter(x, y, z, c=z, cmap='plasma', alpha=0.6, s=30)
        ax1.set_xlabel('Вік')
        ax1.set_ylabel('Середній рейтинг')
        ax1.set_zlabel('Кількість оцінок')
        ax1.set_title('Профіль користувачів')
        
        ax2 = fig.add_subplot(222, projection='3d')
        time_analysis = merged_data.groupby(['rating_year', 'rating_quarter']).agg({
            'userId': 'count'
        }).reset_index()
        
        x = time_analysis['rating_year']
        y = time_analysis['rating_quarter']
        z = time_analysis['userId']
        
        colors = plt.cm.viridis(z / max(z))
        ax2.bar3d(x, y, np.zeros(len(z)), 
                 dx=0.5, dy=0.5, dz=z,
                 color=colors, alpha=0.7)
        ax2.set_xlabel('Рік')
        ax2.set_ylabel('Квартал')
        ax2.set_zlabel('Оцінки')
        ax2.set_title('Активність по часу')
        
        ax3 = fig.add_subplot(223)
        monthly_trends = merged_data.groupby('rating_month').agg({
            'userId': 'count',
            'rating': 'mean'
        })
        ax3.bar(monthly_trends.index, monthly_trends['userId'], alpha=0.7, color='skyblue')
        ax3.set_xlabel('Місяць')
        ax3.set_ylabel('Кількість оцінок')
        ax3.set_title('Активність по місяцях')
        ax3.grid(True, alpha=0.3)
        
        ax4 = fig.add_subplot(224)
        genre_ratings = merged_data.groupby('main_genre').agg({
            'rating': 'mean',
            'userId': 'count'
        }).nlargest(8, 'userId')
        
        x_pos = np.arange(len(genre_ratings))
        ax4.bar(x_pos, genre_ratings['rating'], alpha=0.7, color='lightcoral')
        ax4.set_xlabel('Жанр')
        ax4.set_ylabel('Середній рейтинг')
        ax4.set_title('Рейтинги жанрів')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(genre_ratings.index, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        
        plt.subplots_adjust(left=0.06, right=0.96, top=0.92, bottom=0.08, wspace=0.25, hspace=0.35)
        plt.savefig('media/3d_analysis_comprehensive.png', dpi=200, bbox_inches='tight')
        plt.show()
    
    def generate_statistics_report(self):
        merged_data = self.ratings_df.merge(self.movies_df, on='movieId').merge(self.users_df, on='userId')
        
        report = {
            'Загальна статистика': {
                'Кількість фільмів': len(self.movies_df),
                'Кількість користувачів': len(self.users_df),
                'Кількість оцінок': len(self.ratings_df),
                'Період даних': f"{merged_data['date'].min().strftime('%Y-%m-%d')} - {merged_data['date'].max().strftime('%Y-%m-%d')}",
                'Середній рейтинг': f"{merged_data['rating'].mean():.2f}",
                'Унікальних жанрів': merged_data['main_genre'].nunique()
            },
            'Демографічна статистика': {
                'Середній вік користувачів': f"{self.users_df['age'].mean():.1f}",
                'Розподіл за статтю': dict(self.users_df['gender'].value_counts()),
                'Найпопулярніші професії': dict(self.users_df['occupation'].value_counts().head(5))
            },
            'Аналіз жанрів': {
                'Найпопулярніші жанри': dict(merged_data['main_genre'].value_counts().head(5)),
                'Жанри з найвищим рейтингом': dict(merged_data.groupby('main_genre')['rating'].mean().nlargest(5).round(2)),
                'Жанри з найнижчим рейтингом': dict(merged_data.groupby('main_genre')['rating'].mean().nsmallest(3).round(2))
            },
            'Часовий аналіз': {
                'Найактивніший рік': int(merged_data['rating_year'].mode()[0]),
                'Найактивніший місяць': int(merged_data['rating_month'].mode()[0]),
                'Роки в даних': list(sorted(merged_data['rating_year'].unique())),
                'Загальний період': f"{merged_data['rating_year'].max() - merged_data['rating_year'].min() + 1} років"
            }
        }
        
        print("СТАТИСТИЧНИЙ ЗВІТ:")
        print("=" * 50)
        for category, stats in report.items():
            print(f"\n{category}:")
            print("-" * 30)
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        return report

def main():
    analyzer = MovieRecommendationPivotAnalysis()
    
    analyzer.load_data()
    
    analyzer.create_pivot_tables()
    
    analyzer.create_matlab_style_plots()
    
    analyzer.generate_statistics_report()
    
    print("\nАНАЛІЗ УСПІШНО ЗАВЕРШЕНО!")

if __name__ == "__main__":
    main()