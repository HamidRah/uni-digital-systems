# models.py
from enum import auto
from numpy import average
from sqlalchemy import func
from .extensions import db
from flask_login import UserMixin
from sqlalchemy.ext.hybrid import hybrid_property

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id = db.Column('userId', db.Integer, primary_key=True, nullable=False, autoincrement=True)
    email = db.Column(db.String(255), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)

class Movie(db.Model):
    __tablename__ = 'movies'
    movieId = db.Column(db.Integer, primary_key=True)
    tmdbId = db.Column(db.Integer)
    title = db.Column(db.String(255))
    overview = db.Column(db.Text)
    popularity = db.Column(db.Float)
    release_date = db.Column(db.Date)
    poster_path = db.Column(db.String(255))
    average_rating = db.Column(db.Float)

    genres = db.relationship('Genre', secondary='movie_genres', back_populates='movies')
    tag_scores = db.relationship('TagScore', back_populates='movie')

    @staticmethod
    def fetch_movie_details(movie_ids):
        movies = []
        for movie_id in movie_ids:
            movie = Movie.query.get(movie_id)
            if movie:
                movies.append({
                    'movieId': movie.movieId,
                    'title': movie.title,
                    'overview': movie.overview,
                    'release_date': movie.release_date.strftime('%Y-%m-%d'),
                    'poster_path': movie.poster_path,
                    'genres': ', '.join([genre.genreName for genre in movie.genres]),
                    'average_rating': f"{movie.average_rating:.1f}" if movie.average_rating is not None else "Not Rated",
                })
        return movies


    @hybrid_property
    def get_year(self):
        return self.release_date.strftime('%Y')  # Python-side property, not used in queries

    @get_year.expression
    def year(cls):
        return func.year(cls.release_date)  # Use MySQL YEAR() function for queries

class Genre(db.Model):
    __tablename__ = 'genres'
    genreId = db.Column(db.Integer, primary_key=True, autoincrement=True)
    genreName = db.Column(db.String(50))
    movies = db.relationship('Movie', secondary='movie_genres', back_populates='genres')

class MovieGenres(db.Model):
    __tablename__ = 'movie_genres'
    movieId = db.Column(db.Integer, db.ForeignKey('movies.movieId'), primary_key=True)
    genreId = db.Column(db.Integer, db.ForeignKey('genres.genreId'), primary_key=True)

class Rating(db.Model):
    __tablename__ = 'ratings'
    userId = db.Column(db.Integer, db.ForeignKey('users.userId'), primary_key=True)
    movieId = db.Column(db.Integer, db.ForeignKey('movies.movieId'), primary_key=True)
    rating = db.Column(db.Float)

    user = db.relationship('User', backref='ratings', lazy=True)
    movie = db.relationship('Movie', backref='ratings', lazy=True)

class Tag(db.Model):
    __tablename__ = 'tags'
    tagId = db.Column(db.Integer, primary_key=True)
    tag = db.Column(db.String(255))

    tag_scores = db.relationship('TagScore', back_populates='tag')

class TagScore(db.Model):
    __tablename__ = 'tag_scores'
    movieId = db.Column(db.Integer, db.ForeignKey('movies.movieId'), primary_key=True)
    tagId = db.Column(db.Integer, db.ForeignKey('tags.tagId'), primary_key=True)
    relevance = db.Column(db.Float)

    movie = db.relationship('Movie', back_populates='tag_scores')
    tag = db.relationship('Tag', back_populates='tag_scores')

