CREATE DATABASE IF NOT EXISTS moviemanager_db;
USE moviemanager_db;

DROP TABLE IF EXISTS movie_genres;
DROP TABLE IF EXISTS genres;
DROP TABLE IF EXISTS movies;

DROP TABLE IF EXISTS ratings;

CREATE TABLE movies (
    movieId INT PRIMARY KEY,
    tmdbId INT UNIQUE NOT NULL,
    title VARCHAR(255),
    overview TEXT,
    popularity DECIMAL(5,2),
    release_date DATE,
    poster_path VARCHAR(255),
    average_rating DECIMAL(3,2),
    num_ratings INT
);

CREATE TABLE genres (
    genreId INT AUTO_INCREMENT PRIMARY KEY,
    genreName VARCHAR(50)
);

CREATE TABLE movie_genres (
    movieId INT,
    genreId INT,
    PRIMARY KEY (movieId, genreId),
    FOREIGN KEY (movieId) REFERENCES movies(movieId),
    FOREIGN KEY (genreId) REFERENCES genres(genreId)
);

CREATE TABLE users (
    userId INT AUTO_INCREMENT PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL
);

CREATE TABLE ratings (
    userId INT,
    movieId INT,
    rating DECIMAL(2,1),
    PRIMARY KEY (userId, movieId),
    FOREIGN KEY (userId) REFERENCES users(userId) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (movieId) REFERENCES movies(movieId) ON DELETE CASCADE ON UPDATE CASCADE
);

CREATE TABLE tags (
    tagId INT PRIMARY KEY,
    tag VARCHAR(255)
);

CREATE TABLE tag_scores (
    movieId INT,
    tagId INT,
    relevance DECIMAL(10,9),
    PRIMARY KEY (movieId, tagId),
    FOREIGN KEY (movieId) REFERENCES movies(movieId) ON DELETE CASCADE ON UPDATE CASCADE,
    FOREIGN KEY (tagId) REFERENCES tags(tagId) ON DELETE CASCADE ON UPDATE CASCADE
);


SET GLOBAL autocommit=0; -- Disable autocommit
SET GLOBAL unique_checks=0; -- Disable unique checks
SET GLOBAL foreign_key_checks=0; -- Disable foreign key checks
-- Insert data here



SET GLOBAL foreign_key_checks=1; -- Re-enable foreign key checks
SET GLOBAL unique_checks=1; -- Re-enable unique checks
COMMIT; -- Commit the transaction
SET GLOBAL autocommit=1; -- Re-enable autocommit

-- ALTER TABLE users MODIFY email VARCHAR(255) NOT NULL;
-- ALTER TABLE users MODIFY password_hash VARCHAR(255) NOT NULL;


SET GLOBAL max_allowed_packet=104857600; -- Sets it to 100MB

-- Allow local file loading
SET GLOBAL local_infile=1;

SHOW VARIABLES LIKE 'log_error';

SHOW WARNINGS;

LOAD DATA LOCAL INFILE 'D:\\UWE\\UWECompSci\\Year3\\DSP\\movieManager\\data\\flask_data\\filtered_ratings_chunk_9.csv'
INTO TABLE ratings
FIELDS TERMINATED BY ',' 
LINES TERMINATED BY '\r\n'
IGNORE 1 LINES
(userId, movieId, rating);

-- Disable keys in ratings table for faster insert
ALTER TABLE ratings DISABLE KEYS;

ALTER TABLE ratings ENABLE KEYS;


