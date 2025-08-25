from flask import Flask, render_template, request, redirect, send_file, session, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import pickle
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer

#////////////////////////////////////////////////////////////////

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, UserMixin, current_user
from werkzeug.security import check_password_hash



from flask import render_template, request, redirect, url_for, flash
from sqlalchemy.exc import IntegrityError

import pickle
import numpy as np
import re
#////////////////////////////////////////////////////////////////

import os
from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy



#///////////////////////////////////////////

from flask import Flask, render_template, request, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import pickle
import difflib
import numpy as np
from numpy.linalg import norm
from sklearn.feature_extraction.text import TfidfVectorizer


import requests

import os
from flask import send_from_directory

#///////////////////////////////////////////////

app = Flask(__name__)
app.secret_key = 'your-secret-key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/finalbook'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False




#///////////////////////////payapl

# PayPal configuration (Replace with your actual PayPal sandbox credentials)
PAYPAL_CLIENT_ID = 'ARWR6KW4BgkAjWgh4E_3a7F75NMamkkLJGLP4_IGebbUr8qP3Zp39bDvdpgYlsbizrsSd1ZXPK3hkjiB'
PAYPAL_CLIENT_SECRET = 'EDqoEFe7zsnN2XfrYQBysfTsVU9TnCM6RyriUSLkZeo9thiFeaPp0FfHcN5AI1-YJjy6697jUo7sLfNL'
PAYPAL_API_URL = 'https://api-m.sandbox.paypal.com'  # Use 'https://api-m.paypal.com' for production

# //////////////////////////////////#

# 🔌 DB setup
db = SQLAlchemy(app)


login_manager = LoginManager()
login_manager.login_view = 'login'
login_manager.init_app(app)

# ✅ User model
class User(UserMixin, db.Model):
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    email = db.Column(db.String(255), nullable=False, unique=True)  # new
    role = db.Column(db.String(50), nullable=False, default='user')  # new

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

def is_valid_username(username):
    return re.match(r'^[A-Za-z](?:[A-Za-z ]{1,23})?[A-Za-z]$', username)

def is_valid_password(password):
    return re.match(r'^(?=.*[!@#$%^&*])[A-Za-z\d!@#$%^&*]{8,}$', password)
def is_valid_email(email):
    return re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}$',email)

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form.get('confirm_password')  # assuming you added this field in HTML

        # Field completeness check
        if not username or not email or not password or not confirm_password:
            flash('All fields are required.')
            return redirect(url_for('register'))

        # Username validation
        if not is_valid_username(username):
            flash('Name must be 3–25 characters, only letters and spaces, and contain at least 3 letters.')
            return redirect(url_for('register'))
        
        if not is_valid_email(email):
            flash('Invalid email format.')
            return redirect(url_for('register'))

        # Password strength validation
        if not is_valid_password(password):
            flash('Password must be at least 8 characters long and contain at least one special character (!@#$%^&*).')
            return redirect(url_for('register'))

        # Password confirmation match
        if password != confirm_password:
            flash('Passwords do not match.')
            return redirect(url_for('register'))

        # Check if user exists
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            flash('Username already exists. Try logging in.')
            return redirect(url_for('register'))

        existing_email = User.query.filter_by(email=email).first()
        if existing_email:
            flash('Email already registered. Try logging in.')
            return redirect(url_for('register'))

        new_user = User(username=username, password=password, email=email)

        db.session.add(new_user)
        db.session.commit()

        flash('Registration successful. You can now log in.')
        return redirect(url_for('login'))

    return render_template('register.html')



@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        user = User.query.filter_by(email=email, password=password).first()
        if user:
            login_user(user)
            if user.role == 'admin':
                return redirect(url_for('admin_dashboard'))
            elif user.role == 'user':
                return redirect(url_for('user_dashboard'))
            else:
                return redirect(url_for('recommend'))
        else:
            flash('Invalid email or password')
            return redirect(url_for('login'))

    return render_template('login.html')


# 🚪 Logout
@app.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Logged out successfully.")
    return redirect(url_for('index'))


#///////////////////////////////////////////

# Load initial data
def load_data():
    """Load pickle files and return data"""
    top_50_books = pickle.load(open('data/top_50_books.pkl', 'rb'))
    df = pickle.load(open('data/books.pkl', 'rb'))
    tfidf_matrix = pickle.load(open('data/tfidf_matrix.pkl', 'rb'))
    indices = pickle.load(open('data/indices.pkl', 'rb'))
    return top_50_books, df, tfidf_matrix, indices

# Load data at startup
top_50_books, df, tfidf_matrix, indices = load_data()

# Book model
class Book(db.Model):
    __tablename__ = 'books'

    id = db.Column(db.Integer, primary_key=True)
    author = db.Column(db.String(255), nullable=False)
    description = db.Column(db.Text, nullable=False)
    rating = db.Column(db.Float, nullable=True)
    title = db.Column(db.String(255), nullable=False)
    genres = db.Column(db.String(255), nullable=True)
    image = db.Column(db.String(500), nullable=True)
    pdf = db.Column(db.String(500), nullable=True)
    price = db.Column(db.Float, nullable=False, default=7.99)

# Create table if it doesn't exist
with app.app_context():
    db.create_all()

@app.route('/')
def index():
    return render_template('index.html',
        book_name=list(top_50_books['book_title'].values),
        author=list(top_50_books['book_authors'].values),
        image=list(top_50_books['image_url'].values),
        genres=list(top_50_books['genres'].values),
        book_desc=list(top_50_books['book_desc'].values),
        rating=list(top_50_books['book_rating'].values)
    )

# Cosine similarity function
def manual_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = norm(vec1)
    norm2 = norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return dot_product / (norm1 * norm2)

# Recommend books function
def recommend_books(title, top_n=10):
    # Reload data to get latest updates
    global df, tfidf_matrix, indices
    _, df, tfidf_matrix, indices = load_data()
    
    title = title.lower()
    if title not in indices: # indices=  titlename-> 0
        return []
    
    # Extract the scalar index
    idx = indices[title]
    if hasattr(idx, 'item'): # [0] = 0 if 0 = 0
        idx = idx.item()  # Use .item() if it's a pandas Series
    
    book_vec = tfidf_matrix[idx].toarray().flatten()
    similarities = []

    for i in range(tfidf_matrix.shape[0]):
        if i == idx:
            continue
        other_vec = tfidf_matrix[i].toarray().flatten()
        sim = manual_cosine_similarity(book_vec, other_vec)
        similarities.append((i, sim))

    similarities.sort(key=lambda x: x[1], reverse=True)
    top_indices = [i for i, score in similarities[:top_n]]

    return df.iloc[top_indices].to_dict(orient='records')

@app.route('/recommend', methods=['GET', 'POST'])
def recommend():
    if request.method == 'POST':
        book_title = request.form['book_title']
        recommendations = recommend_books(book_title)
        if not recommendations:
            flash(f"Book '{book_title}' not found. Please check the title and try again.", "warning")
            return render_template('recomm.html', books=[], input_title=book_title)
        return render_template('recomm.html', books=recommendations, input_title=book_title)
    return render_template('recomm.html', books=[], input_title=None)




#///////////////////////////////////////////////////////////////
@app.route('/admin')
@login_required
def admin_dashboard():
    if current_user.role != 'admin':
        flash('Unauthorized access.')
        return redirect(url_for('index'))
    return render_template('admin/adminindex.html',
        book_name=list(top_50_books['book_title'].values),
        author=list(top_50_books['book_authors'].values),
        image=list(top_50_books['image_url'].values),
        genres=list(top_50_books['genres'].values),
        book_desc=list(top_50_books['book_desc'].values),
        rating=list(top_50_books['book_rating'].values)
    )

@app.route('/admin/recommend', methods=['GET', 'POST'])
@login_required
def recommend_admin():
    if request.method == 'POST':
        book_title = request.form['book_title']
        recommendations = recommend_books(book_title)
        if not recommendations:
            flash(f"Book '{book_title}' not found. Please check the title and try again.", "warning")
            return render_template('admin/adminrecomm.html', books=[], input_title=book_title)
        return render_template('admin/adminrecomm.html', books=recommendations, input_title=book_title)
    return render_template('admin/adminrecomm.html', books=[], input_title=None)

@app.route('/admin/addbook', methods=['GET', 'POST'])
@login_required
def add_book():
    global df, tfidf_matrix, indices, top_50_books
    
    if request.method == 'POST':
        # Get form data
        title = request.form['book_title']
        authors = request.form['book_authors']
        rating = request.form.get('book_rating')
        rating = float(rating) if rating else None
        genres = request.form.get('genres')
        desc = request.form['book_desc']
        image_url = request.form.get('image_url')
        price = float(request.form.get('price', 7.99))

        # Check if book exists
        if Book.query.filter_by(title=title).first():
            flash("Book already exists in database!", "warning")
            return redirect(url_for('add_book'))

        try:
            # Save to database
            new_book = Book(
                author=authors,
                description=desc,
                rating=rating,
                title=title,
                genres=genres,
                image=image_url,
            )
            db.session.add(new_book)
            db.session.commit()

            # Load current data
            df = pickle.load(open('data/books.pkl', 'rb'))

            # Create new row
            new_row = pd.DataFrame([{
                'book_title': title,
                'book_authors': authors,
                'book_rating': rating,
                'genres': genres,
                'book_desc': desc,
                'image_url': image_url
            }])

            def clean_text(text):
                return str(text).lower().replace('\n', ' ').replace('\r', '')

            # Add content column to new row
            new_row['content'] = (
                new_row['book_title'].apply(clean_text) + ' ' +
                new_row['genres'].apply(clean_text) + ' ' +
                new_row['book_desc'].apply(clean_text)
            )

            # Concatenate with existing data
            df = pd.concat([df, new_row], ignore_index=True)
            df.drop_duplicates(subset='book_title', keep='first', inplace=True)
            df.dropna(subset=['book_title', 'book_desc', 'genres'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Recreate TF-IDF matrix with all data
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = tfidf.fit_transform(df['content'])
            
            # Recreate indices mapping
            indices = pd.Series(df.index, index=df['book_title'].str.lower())

            # Save updated data to pickle files
            pickle.dump(df, open('data/books.pkl', 'wb'))
            pickle.dump(tfidf_matrix, open('data/tfidf_matrix.pkl', 'wb'))
            pickle.dump(indices, open('data/indices.pkl', 'wb'))

            # Reload global variables
            top_50_books, df, tfidf_matrix, indices = load_data()

            flash("Book added successfully!", "success")
            return redirect(url_for('add_book'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error adding book: {str(e)}", "error")
            return redirect(url_for('add_book'))

    return render_template('admin/adminaddbook.html')



#/////////////////////////////////////////////////////////////////
@app.route('/admin/updatebook', methods=['GET', 'POST'])
def update_book():
    global df, tfidf_matrix, indices, top_50_books
    
    if request.method == 'POST':
        # Get form data
        old_title = request.form['old_book_title']
        new_title = request.form.get('book_title', '')
        authors = request.form.get('book_authors', '')
        rating = request.form.get('book_rating', '')
        rating = float(rating) if rating else None
        genres = request.form.get('genres', '')
        desc = request.form.get('book_desc', '')
        image_url = request.form.get('image_url', '')
        pdf_url = request.form.get('pdf_url', '')
        price = float(request.form.get('price', 7.99))

        # Check if book exists in pickle dataset
        df = pickle.load(open('data/books.pkl', 'rb'))
        book_idx = df[df['book_title'].str.lower() == old_title.lower()].index
        book_in_pickle = not book_idx.empty

        if not book_in_pickle:
            flash(f"Book '{old_title}' not found in dataset!", "warning")
            return redirect(url_for('update_book'))

        try:
            # Check if book exists in database, create if not
            book = Book.query.filter_by(title=old_title).first()
            if not book:
                book_data = df.iloc[book_idx[0]]
                book = Book(
                    author=str(book_data['book_authors']) or 'Unknown Author',
                    description=str(book_data['book_desc']) or 'No description available',
                    rating=book_data['book_rating'] if pd.notnull(book_data['book_rating']) else None,
                    title=book_data['book_title'],
                    genres=str(book_data['genres']) if pd.notnull(book_data['genres']) else None,
                    image=str(book_data['image_url']) if pd.notnull(book_data['image_url']) else None,
                    pdf=None,
                    price=7.99
                )
                db.session.add(book)
                db.session.commit()

            # Update database with provided fields
            book.title = new_title if new_title else book.title
            book.author = authors if authors else book.author
            book.description = desc if desc else book.description
            book.rating = rating if rating is not None else book.rating
            book.genres = genres if genres else book.genres
            book.image = image_url if image_url else book.image
            book.pdf = pdf_url if pdf_url else book.pdf
            book.price = price
            db.session.commit()

            # Update DataFrame
            book_idx = book_idx[0]
            df.loc[book_idx, 'book_title'] = new_title if new_title else df.loc[book_idx, 'book_title']
            df.loc[book_idx, 'book_authors'] = authors if authors else df.loc[book_idx, 'book_authors']
            df.loc[book_idx, 'book_rating'] = rating if rating is not None else df.loc[book_idx, 'book_rating']
            df.loc[book_idx, 'genres'] = genres if genres else df.loc[book_idx, 'genres']
            df.loc[book_idx, 'book_desc'] = desc if desc else df.loc[book_idx, 'book_desc']
            df.loc[book_idx, 'image_url'] = image_url if image_url else df.loc[book_idx, 'image_url']

            def clean_text(text):
                return str(text).lower().replace('\n', ' ').replace('\r', '')

            # Update content column
            df['content'] = (
                df['book_title'].apply(clean_text) + ' ' +
                df['genres'].apply(clean_text) + ' ' +
                df['book_desc'].apply(clean_text)
            )

            # Remove duplicates and nulls
            df.drop_duplicates(subset='book_title', keep='first', inplace=True)
            df.dropna(subset=['book_title', 'book_desc', 'genres'], inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Recreate TF-IDF matrix with all data
            tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
            tfidf_matrix = tfidf.fit_transform(df['content'])
            
            # Recreate indices mapping
            indices = pd.Series(df.index, index=df['book_title'].str.lower())

            # Save updated data to pickle files
            pickle.dump(df, open('data/books.pkl', 'wb'))
            pickle.dump(tfidf_matrix, open('data/tfidf_matrix.pkl', 'wb'))
            pickle.dump(indices, open('data/indices.pkl', 'wb'))

            # Reload global variables
            top_50_books, df, tfidf_matrix, indices = load_data()

            flash("Book updated successfully!", "success")
            return redirect(url_for('update_book'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error updating book: {str(e)}", "error")
            return redirect(url_for('update_book'))

    # For GET request, handle book selection for pre-filling form
    selected_book = None
    if request.args.get('old_book_title'):
        # Check pickle dataset first
        df = pickle.load(open('data/books.pkl', 'rb'))
        book_idx = df[df['book_title'].str.lower() == request.args.get('old_book_title').lower()].index
        if not book_idx.empty:
            book_data = df.iloc[book_idx[0]]
            selected_book = Book(
                title=book_data['book_title'],
                author=str(book_data['book_authors']) or 'Unknown Author',
                description=str(book_data['book_desc']) or 'No description available',
                rating=book_data['book_rating'] if pd.notnull(book_data['book_rating']) else None,
                genres=str(book_data['genres']) if pd.notnull(book_data['genres']) else None,
                image=str(book_data['image_url']) if pd.notnull(book_data['image_url']) else None,
                pdf=None,
                price=7.99
            )
        else:
            flash(f"Book '{request.args.get('old_book_title')}' not found in dataset. Please enter a valid title.", "warning")

    return render_template('/admin/adminupdatebook.html', selected_book=selected_book)


#/////////////////////////////////////////////////////////////////

@app.route('/admin/deletebook', methods=['GET', 'POST'])
def delete_book():
    global df, tfidf_matrix, indices, top_50_books
    
    if request.method == 'POST':
        # Get form data
        title = request.form['book_title']

        # Check if book exists in pickle dataset
        df = pickle.load(open('data/books.pkl', 'rb'))
        book_idx = df[df['book_title'].str.lower() == title.lower()].index
        book_in_pickle = not book_idx.empty

        # Check if book exists in database
        book = Book.query.filter_by(title=title).first()
        book_in_db = book is not None

        if not book_in_pickle and not book_in_db:
            flash(f"Book '{title}' not found in dataset or database!", "warning")
            return redirect(url_for('delete_book'))

        try:
            # Delete from database if it exists
            if book_in_db:
                db.session.delete(book)
                db.session.commit()

            # Delete from DataFrame if it exists
            if book_in_pickle:
                df = df.drop(book_idx)
                df.reset_index(drop=True, inplace=True)

                def clean_text(text):
                    return str(text).lower().replace('\n', ' ').replace('\r', '')

                # Update content column
                df['content'] = (
                    df['book_title'].apply(clean_text) + ' ' +
                    df['genres'].apply(clean_text) + ' ' +
                    df['book_desc'].apply(clean_text)
                )

                # Remove duplicates and nulls
                df.drop_duplicates(subset='book_title', keep='first', inplace=True)
                df.dropna(subset=['book_title', 'book_desc', 'genres'], inplace=True)
                df.reset_index(drop=True, inplace=True)

                # Recreate TF-IDF matrix with updated data
                tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
                tfidf_matrix = tfidf.fit_transform(df['content'])
                
                # Recreate indices mapping
                indices = pd.Series(df.index, index=df['book_title'].str.lower())

                # Save updated data to pickle files
                pickle.dump(df, open('data/books.pkl', 'wb'))
                pickle.dump(tfidf_matrix, open('data/tfidf_matrix.pkl', 'wb'))
                pickle.dump(indices, open('data/indices.pkl', 'wb'))

                # Reload global variables
                top_50_books, df, tfidf_matrix, indices = load_data()

            flash(f"Book '{title}' deleted successfully!", "success")
            return redirect(url_for('delete_book'))

        except Exception as e:
            db.session.rollback()
            flash(f"Error deleting book: {str(e)}", "error")
            return redirect(url_for('delete_book'))

    return render_template('/admin/admindeletebook.html')
#/////////////////////////////////////////////////////////////////



#//////////////////////pdffff
# Upload folder config
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Allowed file extensions for upload
ALLOWED_EXTENSIONS = {'pdf'}

class File(db.Model):
    __tablename__ = 'files'  # Use your actual table name

    id = db.Column(db.Integer, primary_key=True)
    book_name = db.Column(db.String(255), nullable=False)
    author = db.Column(db.String(255), nullable=False)
    pdf_filename = db.Column(db.String(255), nullable=False)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/admin/addpdf', methods=['GET', 'POST'])
def add_pdf():
    if request.method == 'POST':
        book_name = request.form.get('book_name')
        author = request.form.get('author')
        file = request.files.get('pdf')

        if not book_name or not author or not file:
            flash('Please fill all fields and upload a PDF.')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only PDF files are allowed.')
            return redirect(request.url)

        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        new_file = File(book_name=book_name, author=author, pdf_filename=filename)
        db.session.add(new_file)
        db.session.commit()

        flash('Book added successfully!')
        return redirect(url_for('add_pdf'))

    return render_template('admin/adminaddpdf.html')


#/////////////////////////


#//////////////////////////////////////Mnange user
@app.route('/admin/users')
@login_required
def admin_users():
    if current_user.role != 'admin':
        flash("Access denied.")
        return redirect(url_for('index'))
    users = User.query.all()
    return render_template('admin/adminuser.html', users=users)


@app.route('/admin/users/edit/<int:user_id>', methods=['GET', 'POST'])
@login_required
def edit_user(user_id):
    if current_user.role != 'admin':
        flash("Access denied.")
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)
    errors = {}

    if request.method == 'POST':
        username = request.form['username'].strip()
        email = request.form['email'].strip()
        role = request.form['role'].strip().lower()

        # Validation
        if (
            not username or
            not re.match(r'^[A-Za-z](?:[A-Za-z ]{1,23})?[A-Za-z]$', username) or
            sum(c.isalpha() for c in username) < 3):
            errors['username'] = 'Name must be 3–25 characters, only letters and spaces, and contain at least 3 letters.'

        if not email or not re.match(r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[A-Za-z]{2,}$', email):
            errors['email'] = 'Invalid email format.'

        if role not in ['admin', 'user']:
            errors['role'] = 'Role must be either "admin" or "user".'

        # Check for duplicate username/email
        existing_user = User.query.filter(
            ((User.username == username) | (User.email == email)) & (User.id != user.id)
        ).first()
        if existing_user:
            errors['general'] = 'Username or email already exists.'

        # If no errors, update
        if not errors:
            user.username = username
            user.email = email
            user.role = role

            try:
                db.session.commit()
                flash("User updated successfully.")
                return redirect(url_for('admin_users'))
            except Exception as e:
                db.session.rollback()
                errors['general'] = 'An error occurred while updating the user.'

    return render_template('admin/edit_user.html', user=user, errors=errors)

@app.route('/admin/users/delete/<int:user_id>', methods=['POST'])
@login_required
def delete_user(user_id):
    if current_user.role != 'admin':
        flash("Access denied.")
        return redirect(url_for('index'))

    user = User.query.get_or_404(user_id)
    db.session.delete(user)
    db.session.commit()
    flash("User deleted successfully.")
    return redirect(url_for('admin_users'))

#//////////////////////////////////////////

#//////////////////////////////////////////

# ---------------------------
# UPDATE PDF
# ---------------------------
@app.route('/admin/updatepdf/<int:id>', methods=['GET', 'POST'])
def update_pdf(id):
    file_record = File.query.get_or_404(id)

    if request.method == 'POST':
        book_name = request.form.get('book_name')
        author = request.form.get('author')
        new_file = request.files.get('pdf')

        if not book_name or not author:
            flash('Book name and author are required.')
            return redirect(request.url)

        file_record.book_name = book_name
        file_record.author = author

        if new_file and allowed_file(new_file.filename):
            # Remove old file if exists
            old_path = os.path.join(app.config['UPLOAD_FOLDER'], file_record.pdf_filename)
            if os.path.exists(old_path):
                os.remove(old_path)

            filename = new_file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            new_file.save(file_path)
            file_record.pdf_filename = filename
        elif new_file:
            flash('Only PDF files are allowed.')
            return redirect(request.url)

        db.session.commit()
        flash('Book updated successfully!')
        return redirect(url_for('list_pdfs'))

    return render_template('admin/adminupdatepdf.html', file=file_record)


# ---------------------------
# DELETE PDF
# ---------------------------
@app.route('/admin/deletepdf/<int:id>', methods=['POST'])
def delete_pdf(id):
    file_record = File.query.get_or_404(id)

    # Remove file from storage
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_record.pdf_filename)
    if os.path.exists(file_path):
        os.remove(file_path)

    # Remove from database
    db.session.delete(file_record)
    db.session.commit()

    flash('Book deleted successfully!')
    return redirect(url_for('list_pdfs'))


# ---------------------------
# LIST ALL PDFs (for admin view)
# ---------------------------
@app.route('/admin/pdfs')
def list_pdfs():
    files = File.query.all()
    return render_template('admin/adminlistpdf.html', files=files)

#/////////////////////////////////////////////////////////////////


#/////////////////////////////////////////////////////////////////
# @app.route('/user')
# @login_required
# def user_dashboard():
#     if current_user.role != 'user':
#         flash('Unauthorized access.')
#         return redirect(url_for('index'))
#     return render_template('user/userindex.html',
#         book_name=list(top_50_books['book_title'].values),
#         author=list(top_50_books['book_authors'].values),
#         image=list(top_50_books['image_url'].values),
#         genres=list(top_50_books['genres'].values),
#         book_desc=list(top_50_books['book_desc'].values),
#         rating=list(top_50_books['book_rating'].values)
#     )

# @app.route('/user/recommend', methods=['GET', 'POST'])
# @login_required
# def recommend_user():
#     if request.method == 'POST':
#         book_title = request.form['book_title']
#         recommendations = recommend_books(book_title)
#         if not recommendations:
#             flash(f"Book '{book_title}' not found. Please check the title and try again.", "warning")
#             return render_template('user/userrecomm.html', books=[], input_title=book_title)
#         return render_template('user/userrecomm.html', books=recommendations, input_title=book_title)
#     return render_template('user/userrecomm.html', books=[], input_title=None)





#////////////////////////////////////////////////////////////////////

#//////////////////////////////////////////////payment
# Purchase model
class Purchase(db.Model):
    __tablename__ = 'purchases'
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    book_id = db.Column(db.Integer, db.ForeignKey('books.id'), nullable=False)
    paypal_order_id = db.Column(db.String(255), nullable=False)
    purchase_date = db.Column(db.DateTime, nullable=False, default=db.func.now())

# PayPal token helper
def get_paypal_token():
    url = f'{PAYPAL_API_URL}/v1/oauth2/token'
    headers = {'Accept': 'application/json', 'Accept-Language': 'en_US'}
    data = {'grant_type': 'client_credentials'}
    try:
        response = requests.post(url, auth=(PAYPAL_CLIENT_ID, PAYPAL_CLIENT_SECRET), data=data, headers=headers)
        response.raise_for_status()
        print(f"PayPal token response: {response.json()}")
        return response.json()['access_token']
    except Exception as e:
        print(f"PayPal token error: {str(e)}")
        flash(f"PayPal authentication failed: {str(e)}")
        raise

# Create tables
with app.app_context():
    db.create_all()
    # Ensure all books have a non-zero price
    try:
        db.session.query(Book).filter((Book.price <= 0) | (Book.price.is_(None))).update({Book.price: 7.99}, synchronize_session=False)
        db.session.commit()
        print("Updated book prices in database to 7.99 where price was 0 or NULL")
    except Exception as e:
        db.session.rollback()
        print(f"Error updating book prices: {str(e)}")


#////////////////////////////////////////////////////////////////////

# User dashboard
@app.route('/user')
@login_required
def user_dashboard():
    if current_user.role != 'user':
        flash('Unauthorized access.')
        return redirect(url_for('index'))
    
    top_50_books, df, _, _ = load_data()
    
    if top_50_books is None:
        flash('No books available. Please check top_50_books.pkl.', 'error')
        print("Error: top_50_books.pkl failed to load")
        return render_template('user/userindex.html', books=[], purchased_book_ids=[])
    
    if df.empty:
        flash('No book data available. Please check books.pkl.', 'error')
        print("Error: df is empty")
        return render_template('user/userindex.html', books=[], purchased_book_ids=[])
    
    book_objects = []
    
    if isinstance(top_50_books, pd.DataFrame):
        if 'book_title' not in top_50_books.columns:
            flash('Invalid top_50_books.pkl format: missing book_title column.', 'error')
            print("Error: top_50_books DataFrame missing 'book_title' column")
            return render_template('user/userindex.html', books=[], purchased_book_ids=[])
        book_titles = top_50_books['book_title'].tolist()
    else:
        flash('Invalid top_50_books.pkl format: expected DataFrame.', 'error')
        print("Error: top_50_books is not a DataFrame, type:", type(top_50_books))
        return render_template('user/userindex.html', books=[], purchased_book_ids=[])
    
    for book_title in book_titles:
        book_data = df[df['book_title'].str.lower() == str(book_title).lower()]
        if book_data.empty:
            print(f"Warning: Book '{book_title}' not found in books.pkl")
            continue
        
        book_data = book_data.iloc[0]
        db_book = Book.query.filter_by(title=book_data['book_title']).first()
        if not db_book:
            try:
                db_book = Book(
                    title=book_data['book_title'],
                    author=str(book_data.get('book_authors', 'Unknown Author')),
                    description=str(book_data.get('book_desc', 'No description available')),
                    rating=float(book_data['book_rating']) if pd.notnull(book_data.get('book_rating')) else None,
                    genres=str(book_data.get('genres')) if pd.notnull(book_data.get('genres')) else None,
                    image=str(book_data.get('image_url')) if pd.notnull(book_data.get('image_url')) else None,
                    price=7.99
                )
                db.session.add(db_book)
                db.session.commit()
                print(f"Added book '{book_data['book_title']}' to database with price 7.99")
            except Exception as e:
                db.session.rollback()
                print(f"Error adding book '{book_data['book_title']}' to database: {str(e)}")
                continue
        if db_book.price <= 0:
            db_book.price = 7.99
            db.session.commit()
            print(f"Updated book '{db_book.title}' price to 7.99")
        book_objects.append(db_book)
    
    if not book_objects:
        flash('No books available. Please check if titles in top_50_books.pkl match books.pkl.', 'error')
        print("Error: No books matched between top_50_books.pkl and books.pkl")
        return render_template('user/userindex.html', books=[], purchased_book_ids=[])
    
    books = book_objects[:50]
    purchased_book_ids = [p.book_id for p in Purchase.query.filter_by(user_id=current_user.id).all()]
    print("Books IDs:", [book.id for book in books if book])
    print("Purchased Book IDs:", purchased_book_ids)
    return render_template('user/userindex.html', books=books, purchased_book_ids=purchased_book_ids)

# User recommendation
@app.route('/user/recommend', methods=['GET', 'POST'])
@login_required
def recommend_user():
    if current_user.role != 'user':
        flash('Unauthorized access.')
        return redirect(url_for('index'))
    purchased_book_ids = [p.book_id for p in Purchase.query.filter_by(user_id=current_user.id).all()]
    if request.method == 'POST':
        book_title = request.form['book_title']
        recommendations = recommend_books(book_title)
        if not recommendations:
            flash(f"Book '{book_title}' not found. Please check the title and try again.", "warning")
            return render_template('user/userrecomm.html', books=[], book_objects=[], input_title=book_title, purchased_book_ids=purchased_book_ids)
        
        book_objects = []
        for book in recommendations:
            db_book = Book.query.filter_by(title=book['book_title']).first()
            if not db_book:
                db_book = Book(
                    title=book['book_title'],
                    author=book['book_authors'],
                    description=book['book_desc'],
                    rating=book['book_rating'] if pd.notnull(book['book_rating']) else None,
                    genres=book['genres'] if pd.notnull(book['genres']) else None,
                    image=book['image_url'] if pd.notnull(book['image_url']) else None,
                    price=7.99
                )
                db.session.add(db_book)
                db.session.commit()
                print(f"Added recommended book '{book['book_title']}' to database with price 7.99")
            if db_book.price <= 0:
                db_book.price = 7.99
                db.session.commit()
                print(f"Updated recommended book '{db_book.title}' price to 7.99")
            book_objects.append(db_book)
        
        return render_template('user/userrecomm.html', books=recommendations, book_objects=book_objects, input_title=book_title, purchased_book_ids=purchased_book_ids)
    return render_template('user/userrecomm.html', books=[], book_objects=[], input_title=None, purchased_book_ids=purchased_book_ids)

# PayPal checkout
@app.route('/buy/<int:book_id>', methods=['POST'])
@login_required
def buy_book(book_id):
    if current_user.role != 'user':
        flash('Unauthorized access.')
        print("Unauthorized access: User role is not 'user'")
        return redirect(url_for('index'))
    print(f"Processing buy request for book_id: {book_id}, user_id: {current_user.id}")
    book = Book.query.get_or_404(book_id)
    print(f"Book found: {book.title}, Price: {book.price}")
    if book.price <= 0:
        flash('This book is not available for purchase.')
        print("Book price <= 0, redirecting to user_dashboard")
        return redirect(url_for('user_dashboard'))
    try:
        print("Attempting to get PayPal token")
        token = get_paypal_token()
        print("PayPal token obtained")
        headers = {
            'Authorization': f'Bearer {token}',
            'Content-Type': 'application/json'
        }
        payload = {
            'intent': 'CAPTURE',
            'purchase_units': [{
                'amount': {
                    'currency_code': 'USD',
                    'value': str(book.price)
                },
                'description': f'Purchase of {book.title}'
            }],
            'application_context': {
                'return_url': url_for('payment_success', book_id=book_id, _external=True),
                'cancel_url': url_for('user_dashboard', _external=True)
            }
        }
        print(f"Sending PayPal request with payload: {payload}")
        response = requests.post(f'{PAYPAL_API_URL}/v2/checkout/orders', headers=headers, json=payload)
        print(f"PayPal response status: {response.status_code}, body: {response.text}")
        if response.status_code == 201:
            order_data = response.json()
            print(f"PayPal order response: {order_data}")
            session['order_id'] = order_data['id']
            for link in order_data['links']:
                if link['rel'] == 'approve':
                    print(f"Redirecting to PayPal approval URL: {link['href']}")
                    return redirect(link['href'])
            flash('No approval link found in PayPal response.')
            print("No approval link found in PayPal response")
            return redirect(url_for('user_dashboard'))
        flash(f"Error creating PayPal order: {response.text}")
        print(f"PayPal order creation failed: {response.text}")
        return redirect(url_for('user_dashboard'))
    except Exception as e:
        flash(f"Payment error: {str(e)}")
        print(f"Exception in buy_book: {str(e)}")
        return redirect(url_for('user_dashboard'))

# Payment success
@app.route('/payment/success/<int:book_id>')
@login_required
def payment_success(book_id):
    if current_user.role != 'user':
        flash('Unauthorized access.')
        print("Unauthorized access: User role is not 'user'")
        return redirect(url_for('index'))
    order_id = session.get('order_id')
    if not order_id:
        flash('No order found.')
        print("No order found in session")
        return redirect(url_for('user_dashboard'))
    try:
        print(f"Capturing PayPal order: {order_id}")
        token = get_paypal_token()
        headers = {'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'}
        response = requests.post(f'{PAYPAL_API_URL}/v2/checkout/orders/{order_id}/capture', headers=headers)
        print(f"PayPal capture response status: {response.status_code}, body: {response.text}")
        if response.status_code == 201:
            # Verify book exists
            book = Book.query.get_or_404(book_id)
            # Check if purchase already exists to avoid duplicates
            existing_purchase = Purchase.query.filter_by(user_id=current_user.id, book_id=book_id, paypal_order_id=order_id).first()
            if not existing_purchase:
                purchase = Purchase(user_id=current_user.id, book_id=book_id, paypal_order_id=order_id)
                db.session.add(purchase)
                db.session.commit()
                print(f"Purchase recorded for user_id: {current_user.id}, book_id: {book_id}, paypal_order_id: {order_id}")
            else:
                print(f"Purchase already exists for user_id: {current_user.id}, book_id: {book_id}, paypal_order_id: {order_id}")
            flash('Purchase successful!')
            return redirect(url_for('my_library'))
        flash('Payment capture failed.')
        print(f"Payment capture failed: {response.text}")
        return redirect(url_for('user_dashboard'))
    except Exception as e:
        db.session.rollback()
        flash(f"Payment error: {str(e)}")
        print(f"Exception in payment_success: {str(e)}")
        return redirect(url_for('user_dashboard'))

# My Library
@app.route('/user/library')
@login_required
def my_library():
    if current_user.role != 'user':
        flash('Unauthorized access.')
        return redirect(url_for('index'))
    purchases = Purchase.query.filter_by(user_id=current_user.id).all()
    books = [Book.query.get(purchase.book_id) for purchase in purchases]
    # print("Library books:", [book.title for book in books if book])
    # print("Library book IDs:", [book.id for book in books if book])
    # print("Purchase records:", [(p.id, p.book_id, p.paypal_order_id) for p in purchases])
    return render_template('user/userlibrary.html', books=books)

# Download/Read book
@app.route('/download/<int:book_id>')
@login_required
def download_book(book_id):
    if current_user.role != 'user':
        flash('Unauthorized access.')
        print("Unauthorized access: User role is not 'user'")
        return redirect(url_for('index'))
    print(f"Attempting to download book_id: {book_id} for user_id: {current_user.id}")
    purchase = Purchase.query.filter_by(user_id=current_user.id, book_id=book_id).first()
    if not purchase:
        flash('You have not purchased this book.')
        print(f"No purchase found for user_id: {current_user.id}, book_id: {book_id}")
        return redirect(url_for('my_library'))
    file_entry = File.query.filter_by(book_id=book_id).first()
    if not file_entry:
        flash('PDF not available for this book.')
        print(f"No File entry found for book_id: {book_id}")
        return redirect(url_for('my_library'))
    if not os.path.exists(file_entry.pdf_filename):
        flash('PDF file not found on server.')
        print(f"PDF file not found at: {file_entry.pdf_filename}")
        return redirect(url_for('my_library'))
    print(f"Downloading PDF for book_id: {book_id}, path: {file_entry.pdf_filename}")
    return send_file(file_entry.pdf_filename, as_attachment=False)  # Open PDF in browser

# Admin orders
@app.route('/admin/orders')
@login_required
def admin_orders():
    if current_user.role != 'admin':
        flash('Unauthorized access.')
        return redirect(url_for('index'))
    purchases = Purchase.query.all()
    orders = []
    for purchase in purchases:
        user = User.query.get(purchase.user_id)
        book = Book.query.get(purchase.book_id)
        orders.append({
            'order_id': purchase.id,
            'paypal_order_id': purchase.paypal_order_id,
            'username': user.username,
            'book_title': book.title,
            'purchase_date': purchase.purchase_date
        })
    print("Admin orders:", [(o['order_id'], o['book_title']) for o in orders])
    return render_template('admin/adminorders.html', orders=orders)

#////////////////////////




if __name__ == '__main__':
    app.run(debug=True)

