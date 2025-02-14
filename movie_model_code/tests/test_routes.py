# tests/test_routes.py
import re
import unittest
from flask import Flask, url_for
from flask_testing import TestCase
from app.models import Movie, Genre, MovieGenres, Rating, User
from app.app import create_app, db
from datetime import datetime
from bs4 import BeautifulSoup
from werkzeug.security import generate_password_hash
from unittest.mock import patch

class BaseTestCase(TestCase):
    def create_app(self):
        app = create_app('testing')
        return app

    def setUp(self):
        db.create_all()
        # Create test data
        genre = Genre(genreId='1', genreName='Action')
        movie = Movie(movieId='1', title='Test Movie', overview='Test Overview', release_date=datetime.now(), poster_path='test_path', average_rating=5.0)
        movie_genre = MovieGenres(movieId='1', genreId='1')

        db.session.add(genre)
        db.session.add(movie)
        db.session.add(movie_genre)
        db.session.commit()

    def tearDown(self):
        db.session.remove()
        db.drop_all()

class TestIndexRoute(BaseTestCase):

    def test_index(self):

        response = self.client.get('/')
        self.assert200(response)
        self.assertTemplateUsed('index.html')


        # Test POST method with genre filter
        response = self.client.post('/', data={'genres': ['1']})
        self.assert200(response)
        self.assertTemplateUsed('index.html')


        # Test POST method with date filter
        response = self.client.post('/', data={'start_date': '2022-01-01', 'end_date': '2022-12-31'})
        self.assert200(response)
        self.assertTemplateUsed('index.html')


    
class TestSearchRoute(BaseTestCase):
    def test_search_results_no_movie(self):
        response = self.client.get('/search')
        self.assert404(response)
        self.assertTemplateUsed('404.html')
        self.assertIn(b'No search term provided', response.data)

    def test_search_results_post_search(self):
        response = self.client.post('/search', data={'search': 'Test Movie'})
        self.assert200(response)
        self.assertTemplateUsed('search_results.html')

    # def test_search_results_post_genre_filter(self):
    #     response = self.client.post('/search', data={'genres': ['1']})
    #     self.assert200(response)
    #     self.assertTemplateUsed('search_results.html')

    # def test_search_results_post_date_filter(self):
    #     response = self.client.post('/search', data={'start_date': '2022-01-01', 'end_date': '2022-12-31'})
    #     self.assert200(response)
    #     self.assertTemplateUsed('search_results.html')

class TestRegisterRoute(BaseTestCase):
    def test_register_new_user(self):
        with self.client:
            # Register a new user
            response = self.client.post('/register', data={
                'email': 'test@test.com', 
                'password': 'password123'
            })

        self.assertStatus(response, 302)
        user = User.query.filter_by(email='test@test.com').first()
        self.assertIsNotNone(user)

    def test_register_existing_user(self):
        
        # Create an existing user
        existing_user = User(email='test@test.com', password_hash='password123')
        db.session.add(existing_user)
        db.session.commit()

        with self.client:
            # Attempt to register as the existing user
            response = self.client.post('/register', data={
                'email': 'test@test.com', 
                'password': 'password123'
            })

        self.assert200(response)
        self.assertTemplateUsed('register.html')
        self.assertIn(b'Error in email - This email is already used. Please choose a different one.', response.data)

    def test_register_authenticated_user(self):
        # Create a user
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            # Log in the user
            response = self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)
            # Ensure the response was a redirect
            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.location, "Redirection to the dashboard is not occurring as expected")

            # Try to access the register route
            response = self.client.get('/register', follow_redirects=False)
            # Check that the response is a redirect to the dashboard
            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.location, "Redirection to the dashboard when already logged in is not occurring as expected")

class TestLoginRoute(BaseTestCase):
    def test_login_valid_user(self):
        # Create a user
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            # Log in the user
            response = self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)
            # Ensure the response was a redirect
            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.location, "Redirection to the dashboard is not occurring as expected")

    def test_login_invalid_user(self):
        with self.client:
            # Attempt to log in with invalid credentials
            response = self.client.post('/login', data={'email': 'invalid@test.com', 'password': 'password123'}, follow_redirects=False)
            # Ensure the response was not a redirect
            self.assertEqual(response.status_code, 200)
            self.assertIn(b'Login Unsuccessful. Please check email and password', response.data)

    def test_login_authenticated_user(self):
        # Create a user
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            # Log in the user
            response = self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)
            # Ensure the response was a redirect
            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.location, "Redirection to the dashboard is not occurring as expected")

            # Try to access the login route
            response = self.client.get('/login', follow_redirects=False)
            # Check that the response is a redirect to the dashboard
            self.assertEqual(response.status_code, 302)
            self.assertTrue('/dashboard' in response.location, "Redirection to the dashboard when already logged in is not occurring as expected")

class TestRateMovieRoute(BaseTestCase):
    def test_rate_movie_new_rating(self):
        # Create a user and log them in
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

            # Rate a movie
            response = self.client.post('/rate_movie', json={'movieId': '1', 'rating': 5})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {'success': True})

            # Check that the rating was added
            rating = Rating.query.filter_by(userId=user.id, movieId='1').first()
            self.assertIsNotNone(rating)
            self.assertEqual(rating.rating, 5)

    def test_rate_movie_update_rating(self):
        # Create a user and log them in
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

            # Add a rating
            rating = Rating(userId=user.id, movieId='1', rating=3)
            db.session.add(rating)
            db.session.commit()

            # Update the rating
            response = self.client.post('/rate_movie', json={'movieId': '1', 'rating': 5})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {'success': True})

            # Check that the rating was updated
            rating = Rating.query.filter_by(userId=user.id, movieId='1').first()
            self.assertIsNotNone(rating)
            self.assertEqual(rating.rating, 5)

    def test_rate_movie_delete_rating(self):
        # Create a user and log them in
        user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        db.session.add(user)
        db.session.commit()

        with self.client:
            self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

            # Add a rating
            rating = Rating(userId=user.id, movieId='1', rating=3)
            db.session.add(rating)
            db.session.commit()

            # Delete the rating
            response = self.client.post('/rate_movie', json={'movieId': '1', 'rating': 0})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.get_json(), {'success': True})

            # Check that the rating was deleted
            rating = Rating.query.filter_by(userId=user.id, movieId='1').first()
            self.assertIsNone(rating)

class TestRecommendRoute(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        self.movie = Movie.query.filter_by(movieId='1').first()
        db.session.add(self.user)
        db.session.commit()

        # Create a rating for the existing user and movie
        self.rating = Rating(userId=self.user.id, movieId=self.movie.movieId, rating=5)
        db.session.add(self.rating)
        db.session.commit()

    def test_recommend_get(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test GET method
        response = self.client.get('/recommend')
        self.assert200(response)
        self.assertTemplateUsed('movies_recommendations.html')

    def test_recommend_post_genre_filter(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test POST method with genre filter
        response = self.client.post('/recommend', data={'genres': ['1']})
        self.assert200(response)
        self.assertTemplateUsed('movies_recommendations.html')

    def test_recommend_post_date_filter(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test POST method with date filter
        response = self.client.post('/recommend', data={'start_date': '2022-01-01', 'end_date': '2022-12-31'})
        self.assert200(response)
        self.assertTemplateUsed('movies_recommendations.html')

class TestRecommendSingleMovieRoute(BaseTestCase):
    def setUp(self):
        super().setUp()
        self.user = User(email='test@test.com', password_hash=generate_password_hash('password123'))
        self.movie = Movie.query.filter_by(movieId='1').first()
        db.session.add(self.user)
        db.session.commit()

        # Create a rating for the existing user and movie
        self.rating = Rating(userId=self.user.id, movieId=self.movie.movieId, rating=5)
        db.session.add(self.rating)
        db.session.commit()

    def test_predict_single_movie_get(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test GET method without a movie ID
        response = self.client.get('/predict_single_movie')
        self.assert404(response)
        self.assertTemplateUsed('404.html')
        self.assertIn(b'No movie selected', response.data)

    def test_predict_single_movie_post_valid_movie(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test POST method with a valid movie ID
        response = self.client.post('/predict_single_movie', data={'movieId': '1'})
        self.assert200(response)
        self.assertTemplateUsed('movie_visualisation.html')
        self.assertIn(b'Test Movie', response.data)
        self.assertIn(b'Predicted Rating', response.data)
        self.assertIn(b'Influential Genres', response.data)
        self.assertIn(b'Influential Tags', response.data)
        self.assertIn(b'Plotly', response.data)

    def test_predict_single_movie_post_invalid_movie(self):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Test POST method with an invalid movie ID
        response = self.client.post('/predict_single_movie', data={'movieId': '999'}, follow_redirects=True)
        self.assert404(response)
        self.assertTemplateUsed('404.html')
        self.assertIn(b'Movie not found', response.data)

    @patch('app.models.Movie.query')
    def test_predict_single_movie_exception(self, mock_movie_query):
        # Log in a user
        self.client.post('/login', data={'email': 'test@test.com', 'password': 'password123'}, follow_redirects=False)

        # Mock an exception when querying the Movie model
        mock_movie_query.filter_by.side_effect = Exception('Test exception')

        # Test POST method with a valid movie ID that raises an exception
        response = self.client.post('/predict_single_movie', data={'movieId': '1'})
        self.assert404(response)
        self.assertTemplateUsed('404.html')
        self.assertIn(b'An error occurred', response.data)

if __name__ == '__main__':
    unittest.main()