import os
import base64
import hashlib
import json
import shutil
import sqlite3
import sys
import random
import tempfile
import subprocess
import logging

from flask import Blueprint, jsonify, request, current_app, render_template, g, send_file
from typing import Tuple
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename


logger = logging.getLogger(__name__)


DATA_DIR = '/app_data'
DATABASE = f'{DATA_DIR}/app.db'


def init_db(db):
    with open(f'{DATA_DIR}/schema.sql', mode='r') as f:
        db.cursor().executescript(f.read())
    db.commit()

def create() -> Blueprint:
    """
    This function is called by Skiff to create your application's API. You can
    code to initialize things at startup here.
    """
    app = Blueprint("app", __name__)

    # Ensure that the data directories exist
    os.makedirs(os.path.join(DATA_DIR, 'uploads'), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, 's2orc'), exist_ok=True)

    def get_db():
        db = getattr(g, '_database', None)
        if db is None:
            db = g._database = sqlite3.connect(DATABASE)
            init_db(db)
        return db

    # This tells the machinery that powers Skiff (Kubernetes) that your application
    # is ready to receive traffic. Returning a non 200 response code will prevent the
    # application from receiving live requests.
    @app.route("/")
    @app.route("/index")
    def index() -> Tuple[str, int]: # pyright: ignore reportUnusedFunction
        return render_template('index.html')

    @app.route('/upload', methods = ['POST'])
    def upload_file():
        if request.method != 'POST':
            return BadRequest('Only POST requests are supported.')
        f = request.files['file']
        buf = f.read()
        h = hashlib.sha1(buf).hexdigest()

        file_path = os.path.join(DATA_DIR, 'uploads', secure_filename(h + '.pdf'))
        if not os.path.exists(file_path):
            f.seek(0)
            f.save(file_path)

        file_size = os.path.getsize(file_path)
        notification_email = request.form.get('notification_email', '')
        # Save to db
        db = get_db()
        cur = db.cursor()
        cur.execute('insert or ignore into files (pdf_hash, file_size, received_time) values (?, ?, datetime("now"))', (h, file_size))
        cur.execute('insert into work_queue (pdf_hash, status, notification_email, submitted_time) values (?, ?, ?, datetime("now"))', (h, "new", notification_email))
        db.commit()
        user_id = base64.b64encode(notification_email.encode('utf-8')).decode('utf-8').replace('+', '-').replace('/', '_').replace('=', '')
        return jsonify({'file_size': file_size, 'hash': h, 'user_id': user_id})

    @app.route('/pdf/<pdf_hash>', methods = ['GET'])
    def show_pdf(pdf_hash):
        if request.method != 'GET':
            return BadRequest('Only GET requests are supported.')
        file_path = os.path.join(DATA_DIR, 'uploads', secure_filename(pdf_hash + '.pdf'))
        if not os.path.exists(file_path):
            return BadRequest('File not found.')
        return send_file(file_path, mimetype='application/pdf')

    @app.route('/list-results', methods = ['GET'])
    def list_results():
        db = get_db()
        cur = db.execute('select pdf_hash from reviews')
        rows = cur.fetchall()
        hashes = sorted(set([row[0] for row in rows]))

        papers = []
        for pdf_hash in hashes:
            paper = {'title': 'unknown', 'hash': None}
            s2orcf = os.path.join(DATA_DIR, 's2orc', pdf_hash + '.json')
            if os.path.exists(s2orcf):
                with open(s2orcf, 'r') as f:
                    papers2orc = json.load(f)
                paper['title'] = papers2orc['title']
                paper['hash'] = pdf_hash
                papers.append(paper)

        return render_template('list_results.html', papers=papers)

    @app.route('/hold-job/<job_id>', methods = ['POST'])
    def hold_job(job_id):
        if request.method != 'POST':
            return BadRequest('Only POST requests are supported.')
        job_id = int(job_id)
        db = get_db()
        cur = db.cursor()
        cur.execute('update work_queue set status = ? where rowid = ?', ("held", job_id))
        db.commit()
        return jsonify({'status': 'ok'})

    @app.route('/delete-review/<row_id>', methods = ['POST'])
    def delete_review(row_id):
        if request.method != 'POST':
            return BadRequest('Only POST requests are supported.')
        row_id = int(row_id)
        db = get_db()
        cur = db.cursor()
        cur.execute('delete from reviews where rowid = ?', (row_id,))
        db.commit()
        return jsonify({'status': 'ok'})


    @app.route('/result/<pdf_hash>', methods = ['GET'])
    @app.route('/survey/<pdf_hash>/<user_id>', methods = ['GET'])
    def show_result(pdf_hash, user_id="unknown_user"):

        user = {'user_id': user_id}

        paper = {'title': 'unknown', 'hash': pdf_hash}
        s2orcf = os.path.join(DATA_DIR, 's2orc', pdf_hash + '.json')
        if os.path.exists(s2orcf):
            with open(s2orcf, 'r') as f:
                papers2orc = json.load(f)
            paper['title'] = papers2orc['title']

        db = get_db()
        cur = db.execute('select pdf_hash, method, review_json, rowid from reviews where pdf_hash = ?', (pdf_hash,))
        rows = cur.fetchall()
        if len(rows) == 0:
            return render_template('results.html', paper=paper, reviews=[], user=user, show_methods=False)

        # Render the "results" template.  We want an unordered list of the
        # comments in each review_json (which is a list of comments) and the
        # method used to generate the review.
        reviews = []
        for row in rows:
            reviews.append({'name': row[1], "method_id": row[1], 'row_id': row[3], 'comments': json.loads(row[2])})

        # For surveys we show reviews in a random order to the user (for
        # anti-bias).  However, we want the order to be consistent for a given
        # survey, so we use the pdf_hash and user_id to seed the random number
        # generator.
        show_methods = True
        if request.path.startswith('/survey'):
            show_methods = False
            random.seed(pdf_hash + '/' + user_id)
            random.shuffle(reviews)

        return render_template('results.html', reviews=reviews, paper=paper, user=user, show_methods=show_methods)

    @app.route('/survey-submit', methods = ['POST'])
    def save_survey():
        if request.method != 'POST':
            return BadRequest('Only POST requests are supported.')
        data = request.form.to_dict()
        db = get_db()
        db.execute('insert into surveys (pdf_hash, user_id, survey_json, received_timestamp, ip, user_agent) values (?, ?, ?, datetime("now"), ?, ?)', (data['pdf_hash'], data['user_id'], json.dumps(data), request.remote_addr, str(request.user_agent)))
        db.commit()
        return jsonify({'status': 'success'})

    @app.route('/favicon.ico')
    def favicon():
        return send_file('static/favicon.ico')

    return app
