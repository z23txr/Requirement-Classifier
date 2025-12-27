from flask import (
    Flask, render_template, request, redirect, url_for, flash,
    send_file, session
)
from flask_mail import Mail, Message
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import pickle
import json
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO
import base64
from werkzeug.security import generate_password_hash, check_password_hash
from collections import namedtuple

# ------------------ Configuration ------------------
load_dotenv()
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "supersecret")

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

app.config.update(
    MAIL_SERVER='smtp.gmail.com',
    MAIL_PORT=587,
    MAIL_USE_TLS=True,
    MAIL_USERNAME=os.getenv("MAIL_USERNAME"),
    MAIL_PASSWORD=os.getenv("MAIL_PASSWORD"),
    MAIL_DEFAULT_SENDER=os.getenv("MAIL_USERNAME"),
)
mail = Mail(app)

User = namedtuple('User', ['username', 'is_authenticated'])

# ------------------ Load Model ------------------
MODEL_PATH = 'model.pkl'
model = pickle.load(open(MODEL_PATH, 'rb')) if os.path.exists(MODEL_PATH) else None

# ------------------ History ------------------
HISTORY_FILE = 'history.json'

def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

def save_history(history):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

history = load_history()

# ------------------ Users ------------------
USERS_FILE = 'users.json'

def load_users():
    if os.path.exists(USERS_FILE):
        with open(USERS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(users, f, ensure_ascii=False, indent=2)

users = load_users()

# ------------------ Helpers ------------------

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def is_logged_in():
    return 'user' in session

@app.context_processor
def inject_current_user():
    username = session.get('user')
    return dict(current_user=User(username=username, is_authenticated=True)) if username else dict(current_user=User(username=None, is_authenticated=False))

# ------------------ Routes ------------------

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if not is_logged_in():
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('login'))

    text = request.form.get('requirement_text', '').strip()
    if not text:
        flash("⚠️ Please enter a requirement.", "warning")
        return redirect(url_for('index'))

    if not model:
        flash("❌ Model not loaded.", "danger")
        return redirect(url_for('index'))

    prediction = model.predict([text])[0]
    history.append({'requirement': text, 'prediction': prediction})
    save_history(history)

    return render_template('index.html', prediction=prediction)

@app.route('/upload', methods=['POST'])
def upload():
    if not is_logged_in():
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('login'))

    file = request.files.get('file')
    if not file or file.filename == '':
        flash("⚠️ No file selected.", "warning")
        return redirect(url_for('index'))

    if not allowed_file(file.filename):
        flash("❌ Unsupported file type.", "danger")
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        df = pd.read_csv(filepath) if filename.endswith('.csv') else pd.read_excel(filepath)
        if 'requirement' not in df.columns:
            flash("❌ Column 'requirement' not found.", "danger")
            return redirect(url_for('index'))

        df['prediction'] = model.predict(df['requirement'].astype(str))
        for _, row in df.iterrows():
            history.append({'requirement': row['requirement'], 'prediction': row['prediction']})
        save_history(history)

        df.to_csv('categorized_output.csv', index=False)
        flash("✅ Successfully categorized!", "success")

        return render_template(
            'index.html',
            functional=df[df['prediction'] == 'functional']['requirement'].tolist(),
            non_functional=df[df['prediction'] == 'non-functional']['requirement'].tolist()
        )
    except Exception as e:
        flash(f"❌ Error: {e}", "danger")
        return redirect(url_for('index'))

@app.route('/categories')
def categories():
    return render_template('categories.html', history=history)

@app.route('/delete/<int:index>', methods=['POST'])
def delete_history_item(index):
    if not is_logged_in():
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('login'))

    if 0 <= index < len(history):
        history.pop(index)
        save_history(history)
        flash("🗑️ Entry deleted.", "info")
    else:
        flash("❌ Invalid index.", "danger")
    return redirect(url_for('categories'))

@app.route('/download')
def download():
    if not is_logged_in():
        flash("⚠️ Please login first.", "warning")
        return redirect(url_for('login'))

    path = 'categorized_output.csv'
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    flash("❌ No file available to download.", "danger")
    return redirect(url_for('index'))

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact', methods=['GET', 'POST'])
def contact():
    if request.method == 'POST':
        name = request.form['name']
        email = request.form['email']
        message_body = request.form['message']

        msg = Message(
            subject=f"New message from {name}",
            sender=email,
            recipients=[os.getenv("MAIL_USERNAME")],
            body=f"From: {name} <{email}>\n\n{message_body}"
        )

        try:
            mail.send(msg)
            flash("✅ Message sent!", "success")
        except Exception as e:
            print(f"Email error: {e}")
            flash("❌ Could not send message.", "danger")

        return redirect(url_for('contact'))

    return render_template('contact.html')

@app.route('/faq')
def faq():
    faqs = [
        {'question': 'What is this site about?', 'answer': 'It predicts requirement types.'},
        {'question': 'How accurate is the prediction?', 'answer': 'Accuracy depends on the model.'},
    ]
    return render_template('faq.html', faqs=faqs)

@app.route('/graph')
def graph():
    func_count = sum(1 for h in history if h['prediction'] == 'functional')
    nonfunc_count = sum(1 for h in history if h['prediction'] == 'non-functional')
    total = func_count + nonfunc_count

    if total == 0:
        flash("⚠️ No data available to generate the graph.", "warning")
        return redirect(url_for('index'))

    labels = ['Functional', 'Non-Functional']
    sizes = [func_count, nonfunc_count]
    colors = ['#28a745', '#dc3545']

    fig, ax = plt.subplots()
    ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
    ax.axis('equal')

    img = BytesIO()
    plt.savefig(img, format='png', bbox_inches='tight')
    plt.close(fig)
    img.seek(0)
    graph_url = base64.b64encode(img.getvalue()).decode()

    return render_template('graph.html', graph_url=graph_url)

# ---------------- Authentication ----------------

@app.route('/login', methods=['GET', 'POST'])
def login():
    if is_logged_in():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')

        user = users.get(username)
        if user and check_password_hash(user['password'], password):
            session['user'] = username
            flash(f"✅ Welcome back, {username}!", "success")
            return redirect(url_for('index'))
        else:
            flash("❌ Invalid username or password.", "danger")

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if is_logged_in():
        return redirect(url_for('index'))

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '')
        password2 = request.form.get('password2', '')

        if not username or not email or not password or not password2:
            flash("⚠️ All fields are required.", "warning")
            return redirect(url_for('signup'))

        if password != password2:
            flash("❌ Passwords do not match.", "danger")
            return redirect(url_for('signup'))

        if username in users:
            flash("❌ Username already taken.", "danger")
            return redirect(url_for('signup'))

        hashed_pw = generate_password_hash(password)
        users[username] = {"email": email, "password": hashed_pw}
        save_users(users)
        flash("✅ Signup successful! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("👋 Logged out successfully.", "info")
    return redirect(url_for('login'))

# ------------------ Run ------------------

if __name__ == '__main__':
    app.run(debug=True)
