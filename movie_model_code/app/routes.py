# routes.py
from ctypes import alignment
from email import message
from re import search
import re
from flask import jsonify, render_template, request, redirect, url_for, flash, Blueprint, current_app, session
from flask_login import login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

import datetime

from .models import User, Movie, Genre, MovieGenres, Rating, Tag, TagScore
from .forms import LoginForm, RegistrationForm, SearchForm, GenreFilterForm, YearFilterForm
from .extensions import db, login_manager

from sqlalchemy import func, distinct
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import joinedload

import torch

import plotly.express as px
from flask_paginate import Pagination, get_page_parameter


bp = Blueprint('main', __name__, url_prefix='/')

@bp.app_errorhandler(404)
def error_page(e):
    message = f"An error occurred: {str(e)}"
    flash(message, 'error')
    return render_template('404.html', message=message), 404

@bp.app_errorhandler(500)
def internal_error(e):
    flash("Internal server error", 'error')
    return render_template('404.html'), 500


@bp.route('/', methods=['GET', 'POST'])
def index():
    try:
        search_form = SearchForm()
        genre_filter_form = GenreFilterForm()
        genre_filter_form.genres.choices = [(str(genre.genreId), genre.genreName) for genre in Genre.query.all()]
        year_filter_form = YearFilterForm()

        page = request.args.get('page', 1, type=int)  # Get the current page number from the URL query parameter
        per_page = 12  # Number of movies to display per page

        # Initialize filter values
        selected_genres = request.form.getlist('genres')
        selected_start_date = request.form.get('start_date', '')
        selected_end_date = request.form.get('end_date', '')
        sort_order = request.form.get('sort_order', '')

        movies_query = Movie.query.options(joinedload(Movie.genres))

        if request.method == 'POST':
            print(f'sort_order: {sort_order}')
            if selected_genres:
                for genre_id in selected_genres:
                    movies_query = movies_query.filter(Movie.genres.any(Genre.genreId == genre_id))

            if selected_start_date and selected_end_date:
                movies_query = movies_query.filter(Movie.release_date >= datetime.datetime.strptime(selected_start_date, '%Y-%m-%d'), Movie.release_date <= datetime.datetime.strptime(selected_end_date, '%Y-%m-%d'))

            if sort_order == 'average_rating':
                movies_query = movies_query.order_by(Movie.average_rating.desc())
            elif sort_order == 'oldest':
                movies_query = movies_query.order_by(Movie.release_date.asc())
            elif sort_order == 'newest':
                movies_query = movies_query.order_by(Movie.release_date.desc())
            elif sort_order == 'popularity':
                movies_query = movies_query.order_by(Movie.popularity.desc())
            else:
                movies_query = movies_query.order_by(Movie.popularity.desc())
        else:
            movies_query = movies_query.order_by(Movie.popularity.desc())

        movies_pagination = movies_query.paginate(page=page, per_page=per_page)  # Paginate the movies query

        movies_list = []
        for movie in movies_pagination.items:  # Iterate over the movies on the current page
            movie_data = {
                'movieId': movie.movieId,
                'title': movie.title,
                'overview': movie.overview,
                'release_date': movie.release_date.strftime('%Y-%m-%d'),
                'poster_path': movie.poster_path,
                'genres': ', '.join([genre.genreName for genre in movie.genres]),
                'average_rating': f"{movie.average_rating:.1f}" if movie.average_rating is not None else "Not Rated",
            }
            if current_user.is_authenticated:
                rating = Rating.query.filter_by(userId=current_user.id, movieId=movie.movieId).first()
                if rating:
                    movie_data['user_rating'] = rating.rating
            movies_list.append(movie_data)

        return render_template('index.html', movies=movies_list, search_form=search_form, genre_filter_form=genre_filter_form,
                               year_filter_form=year_filter_form, selected_start_date=selected_start_date,
                               selected_end_date=selected_end_date, selected_sort_order=sort_order,
                               pagination=movies_pagination)  # Pass the pagination object to the template
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return error_page(e)


@bp.route('/search', methods=['GET', 'POST'])
def search_results():
    try:
        search_form = SearchForm()
        genre_filter_form = GenreFilterForm()
        genre_filter_form.genres.choices = [(str(genre.genreId), genre.genreName) for genre in Genre.query.all()]
        year_filter_form = YearFilterForm()

        # Retrieve or update the search term in the session
        if request.method == 'POST' and 'search' in request.form:
            session['search_term'] = request.form.get('search', '')
        search_term = session.get('search_term', '')

        selected_genres = request.form.getlist('genres')
        selected_start_date = request.form.get('start_date', '')
        selected_end_date = request.form.get('end_date', '')

        movies_query = Movie.query.options(joinedload(Movie.genres)).order_by(Movie.popularity.desc())

        # Apply search and filters
        if search_term:
            movies_query = movies_query.join(Movie.tag_scores).join(Tag).filter(
                db.or_(
                    Movie.title.ilike(f'%{search_term}%'),
                    Tag.tag.ilike(f'%{search_term}%')
                )
            )
        if selected_genres:
            movies_query = movies_query.join(MovieGenres).join(Genre).filter(Genre.genreId.in_(selected_genres))
        if selected_start_date and selected_end_date:
            movies_query = movies_query.filter(
                Movie.release_date >= datetime.datetime.strptime(selected_start_date, '%Y-%m-%d'), 
                Movie.release_date <= datetime.datetime.strptime(selected_end_date, '%Y-%m-%d')
            )

        movies = movies_query.limit(100).all()
        # Prepare movie data for the template
        movies_list = prepare_movie_data(movies, search_term)

        return render_template('search_results.html', movies=movies_list, search_form=search_form, genre_filter_form=genre_filter_form, year_filter_form=year_filter_form, 
                            query=search_term, selected_genres=selected_genres, selected_start_date=selected_start_date, selected_end_date=selected_end_date)
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return error_page(e)

def prepare_movie_data(movies, search_term):
    # This function prepares the movie data for the template
    movies_list = [{
        'movieId': movie.movieId,
        'title': movie.title,
        'overview': movie.overview,
        'release_date': movie.release_date.strftime('%Y-%m-%d'),
        'poster_path': movie.poster_path,
        'genres': ', '.join([genre.genreName for genre in movie.genres]),
        'tags': ', '.join([tag_score.tag.tag for tag_score in movie.tag_scores]) if search_term else '',
        'average_rating': f"{movie.average_rating:.1f}" if movie.average_rating is not None else "Not Rated",
        'user_rating': ''
    } for movie in movies]
    if current_user.is_authenticated:
        for movie_data in movies_list:
            rating = Rating.query.filter_by(userId=current_user.id, movieId=movie_data['movieId']).first()
            if rating:
                movie_data['user_rating'] = rating.rating
    return movies_list

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


@bp.route('/register', methods=['GET', 'POST'])
def register():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    else:
        form = RegistrationForm()
        if form.validate_on_submit():
            try:
                hashed_password = generate_password_hash(form.password.data)
                user = User(email=form.email.data, password_hash=hashed_password)
                db.session.add(user)
                db.session.commit()
                flash('Your account has been created! You are now able to log in', 'success')
                return redirect(url_for('main.login'))
            except IntegrityError as e:
                print(f'An error occurred during registration: {e}')
                db.session.rollback()  # Roll back the session to a clean state
                flash('This email is already registered.', 'error')
            except Exception as e:
                db.session.rollback()  # Roll back for any other exceptions
                print(f'An unexpected error occurred during registration: {e}')
                flash('An unexpected error occurred during registration.', 'error')
        else:
            # After form submission, if there are errors (including ValidationError)
            for fieldName, errorMessages in form.errors.items():
                for err in errorMessages:
                    # Flash each error message and print to the console
                    flash(f'Error in {fieldName} - {err}', 'error')
                    print(f'Error in {fieldName}: {err}')
        return render_template('register.html', title='Register', form=form)

@bp.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('main.dashboard'))
    else:
        form = LoginForm()
        if form.validate_on_submit():
            user = User.query.filter_by(email=form.email.data).first()
            if user and check_password_hash(user.password_hash, form.password.data):
                login_user(user)
                return redirect(url_for('main.dashboard'))
            else:
                flash('Login Unsuccessful. Please check email and password', 'error')
                print('Login Unsuccessful. Please check email and password')
        return render_template('login.html', form=form)

@bp.route('/dashboard')
@login_required
def dashboard():
    try:
        sort_order = request.args.get('sort', 'asc')
        search_form = SearchForm()

        user_id = current_user.id
        if sort_order == 'asc':
            rated_movies = Rating.query.filter_by(userId=user_id).order_by(Rating.rating.asc()).all()
        else:
            rated_movies = Rating.query.filter_by(userId=user_id).order_by(Rating.rating.desc()).all()

        movie_data = []
        for rating in rated_movies:
            movie = Movie.query.get(rating.movieId)
            movie_data.append({
                'movieId': movie.movieId,
                'title': movie.title,
                'user_rating': rating.rating,
                'release_date': movie.release_date.strftime('%Y-%m-%d'),
                'genres': ', '.join([genre.genreName for genre in movie.genres]),
            })
        return render_template('dashboard.html', rated_movies=movie_data, search_form=search_form, sort_order=sort_order)
    
    except Exception as e:
        return f"An error occurred: {str(e)}"

@bp.route('/logout')
@login_required
def logout():
    logout_user()
    flash('You have been logged out.', 'success')
    return redirect(url_for('main.index'))


@bp.route('/rate_movie', methods=['POST'])
@login_required
def rate_movie():
    data = request.get_json()
    movieId = data.get('movieId', None)
    rating = data.get('rating', None)

    if movieId is None or rating is None:
        return jsonify({'success': False, 'error': 'Missing movieId or rating'}), 400


    existing_rating = Rating.query.filter_by(userId=current_user.id, movieId=movieId).first()

    if existing_rating:
        if rating == 0:
            # If the rating is 0, delete the rating
            db.session.delete(existing_rating)
        else:
            # Update the existing rating
            existing_rating.rating = rating
    else:
        # Create a new rating with the current date
        new_rating = Rating(userId=current_user.id, movieId=movieId, rating=rating)
        db.session.add(new_rating)

    db.session.commit()

    return jsonify({'success': True})

def get_top_100_movie_ids():
    top_movies = Movie.query.order_by(Movie.popularity.desc()).limit(100).all()
    return [movie.movieId for movie in top_movies]



@bp.route('/recommend', methods=['GET', 'POST'])
@login_required
def recommend():
    try:
        search_form = SearchForm()
        genre_filter_form = GenreFilterForm()
        genre_filter_form.genres.choices = [(str(genre.genreId), genre.genreName) for genre in Genre.query.all()]

        year_filter_form = YearFilterForm()

        page = request.args.get(get_page_parameter(), type=int, default=1)  # Get the current page number from the URL query parameter
        per_page = 12  # Number of movies to display per page

        top_movies = Movie.query.order_by(func.random()).limit(100).all()
        movie_dict = {movie.movieId: movie for movie in top_movies}

        # Convert user and movie IDs to tensors
        user_id = current_user.id
        user_id_tensor = torch.tensor([user_id], dtype=torch.long)
        movie_ids_tensor = torch.tensor([movie.movieId for movie in top_movies], dtype=torch.long)

        # Fetch predictions from the model
        model = current_app.config.get('model')
        model.eval()
        # get max movie embedding shape of model


        with torch.no_grad():
            predictions = model.predict_ratings(movie_ids_tensor, user_id_tensor)

        # Create a lookup for predictions and attach details to movies
        for pred in predictions:
            movie = movie_dict.get(pred['movie_id'])
            if movie:
                movie.predicted_rating = pred['predicted_rating']
                movie.influential_tags = ', '.join(pred['tags'])
                movie.influential_genres = ', '.join(pred['genres'])
        
        selected_start_date = ''
        selected_end_date = ''
        sort_order = request.form.get('sort_order', '')

        if request.method == 'POST':
            selected_genres = request.form.getlist('genres')
            selected_start_date = request.form.get('start_date', '')
            selected_end_date = request.form.get('end_date', '')

            # Filter by genre if genres are selected
            if selected_genres:
                selected_genre_ids = [int(genre) for genre in selected_genres]
                top_movies = [movie for movie in top_movies if all(genre_id in (genre.genreId for genre in movie.genres) for genre_id in selected_genre_ids)]

            # Filter by date if start and end dates are selected
            if selected_start_date:
                start_date = datetime.datetime.strptime(selected_start_date, '%Y-%m-%d').date()
                end_date = datetime.datetime.strptime(selected_end_date, '%Y-%m-%d').date() if selected_end_date else datetime.date.today()
                top_movies = [movie for movie in top_movies if movie.release_date and start_date <= movie.release_date <= end_date]

        # Apply sorting order
        if sort_order == 'average_rating':
            top_movies = sorted(top_movies, key=lambda movie: movie.average_rating or 0, reverse=True)
        elif sort_order == 'oldest':
            top_movies = sorted(top_movies, key=lambda movie: movie.release_date or datetime.date.min)
        elif sort_order == 'newest':
            top_movies = sorted(top_movies, key=lambda movie: movie.release_date or datetime.date.min, reverse=True)
        elif sort_order == 'popularity':
            top_movies = sorted(top_movies, key=lambda movie: movie.popularity or 0, reverse=True)
        else:
            top_movies = sorted(top_movies, key=lambda movie: movie.predicted_rating or 0, reverse=True)

        # Prepare movies list for rendering
        movies_list = [{
            'movieId': movie.movieId,
            'title': movie.title,
            'overview': movie.overview,
            'popularity': movie.popularity,
            'release_date': movie.release_date.strftime('%Y-%m-%d') if movie.release_date else 'Unknown',
            'poster_path': movie.poster_path,
            'genres': ', '.join([genre.genreName for genre in movie.genres]),
            'predicted_rating': getattr(movie, 'predicted_rating', 'N/A'),
            'influential_tags': getattr(movie, 'influential_tags', ''),
            'influential_genres': getattr(movie, 'influential_genres', ''),
            'average_rating': f"{movie.average_rating:.1f}" if movie.average_rating is not None else "Not Rated",
            'user_rating': Rating.query.filter_by(userId=current_user.id, movieId=movie.movieId).first().rating if Rating.query.filter_by(userId=current_user.id, movieId=movie.movieId).first() else "Not Rated"
        } for movie in top_movies]

        # Paginate the movies list using flask_paginate
        pagination = Pagination(page=page, per_page=per_page, total=len(movies_list), record_name='movies',
                        css_framework='bootstrap5', link_size='sm',
                        alignment='center', show_single_page=True)
        paginated_movies = movies_list[(page - 1) * per_page:page * per_page]

        return render_template('movies_recommendations.html', movies=paginated_movies, search_form=search_form, genre_filter_form=genre_filter_form, year_filter_form=year_filter_form,
                               selected_start_date=selected_start_date, selected_end_date=selected_end_date, selected_sort_order=sort_order,
                               pagination=pagination)
    
    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return error_page(e)
    

@bp.route('/predict_single_movie', methods=['GET', 'POST'])
def predict_single_movie():
    try:
        search_form = SearchForm()
        if request.method == 'POST':
            selected_movie_id = int(request.form.get('movieId'))
            movie = Movie.query.filter_by(movieId=selected_movie_id).first()

            if movie:
                user_id = current_user.id if current_user.is_authenticated else None
                user_id_tensor = torch.tensor([user_id], dtype=torch.long) if user_id else None
                movie_id_tensor = torch.tensor([selected_movie_id], dtype=torch.long)
    
                # Fetch predictions from the model
                model = current_app.config.get('model')
                model.eval()
                with torch.no_grad():
                    predictions = model.predict_rating_visual(movie_id_tensor, user_id_tensor)

                movie.predicted_rating = predictions[0]['predicted_rating']

                # Visualize influential tags and genres
                fig_tags = px.bar(x=predictions[0]['tags'], y=[1]*len(predictions[0]['tags']), labels={'x': 'Tag', 'y': 'Influence'}, title="Influential Tags")
                fig_genres = px.bar(x=predictions[0]['genres'], y=[1]*len(predictions[0]['genres']), labels={'x': 'Genre', 'y': 'Influence'}, title="Influential Genres")

                div_tags = fig_tags.to_html(full_html=False)
                div_genres = fig_genres.to_html(full_html=False)

                # Prepare movie details
                movie_details = {
                    'movieId': movie.movieId,
                    'title': movie.title,
                    'overview': movie.overview,
                    'release_date': movie.release_date.strftime('%Y-%m-%d') if movie.release_date else 'Unknown',
                    'poster_path': movie.poster_path,
                    'genres': ', '.join([genre.genreName for genre in movie.genres]),
                    'predicted_rating': getattr(movie, 'predicted_rating', 'N/A'),
                    'influential_tags': getattr(movie, 'influential_tags', ''),
                    'influential_genres': getattr(movie, 'influential_genres', ''),
                    'average_rating': f"{movie.average_rating:.1f}" if movie.average_rating is not None else "Not Rated",
                    'user_rating': Rating.query.filter_by(userId=user_id, movieId=movie.movieId).first().rating if Rating.query.filter_by(userId=user_id, movieId=movie.movieId).first() else "Not Rated"
                }

                return render_template('movie_visualisation.html', movie=movie_details, prediction=predictions[0], plot_div_tags=div_tags, plot_div_genres=div_genres, search_form=search_form)
            else:
                print('Movie not found')
                return error_page('Movie not found')
        else:
            print('No movie selected')
            return error_page('No movie selected')

    except Exception as e:
        flash(f"An error occurred: {str(e)}", 'error')
        return error_page(e)

@bp.route('/user_vs_predicted', methods=['GET', 'POST'])
@login_required
def user_vs_predicted():
    search_form = SearchForm()
    user_id = current_user.id
    # Fetch all ratings made by the user
    user_ratings = Rating.query.filter_by(userId=user_id).all()

    # Get the movie IDs and actual ratings
    movie_ids = [rating.movieId for rating in user_ratings]
    actual_ratings = [rating.rating for rating in user_ratings]

    # Convert movie IDs to tensor
    movie_id_tensor = torch.tensor(movie_ids, dtype=torch.long)

    # Fetch predictions from the model
    model = current_app.config.get('model')
    model.eval()
    with torch.no_grad():
        predictions = model.predict_ratings(movie_id_tensor, torch.tensor([user_id], dtype=torch.long))

    # Extract predicted ratings
    predicted_ratings = [prediction['predicted_rating'] for prediction in predictions]

    # Create scatter plot
    fig = px.scatter(x=actual_ratings, y=predicted_ratings, labels={'x': 'User Ratings', 'y': 'Predicted Ratings'}, title="User Ratings vs Predicted Ratings", template='plotly_dark')
    # Change the marker color to red
    fig.update_traces(marker=dict(color='red'))

    # Style the plot
    fig.update_layout(
        title_font=dict(size=24, color='white'),
        xaxis=dict(
            title='User Ratings',
            gridcolor='darkblue',
            gridwidth=2,
            title_font=dict(size=18, color='white'),
            tickfont=dict(color='white'),
        ),
        yaxis=dict(
            title='Predicted Ratings',
            gridcolor='darkblue',
            gridwidth=2,
            title_font=dict(size=18, color='white'),
            tickfont=dict(color='white'),
        ),
        paper_bgcolor='black',
        plot_bgcolor='black',
    )

    plot_div = fig.to_html(full_html=False)

    return render_template('scatter_plot.html', plot_div=plot_div, search_form=search_form)


