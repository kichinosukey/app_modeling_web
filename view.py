import os
import sys
sys.path.append(os.path.join(os.path.dirname(__name__), '../../'))
import random
from datetime import datetime

import requests
from flask import (
    Blueprint, render_template, request, session, jsonify, g, 
    redirect, flash, send_from_directory, send_file)
from werkzeug.utils import secure_filename

from apps.config import (
    ROOTDIR, FILE_NAME_URL_PREFIX, SESSION_KEY_URL_INDEX, SESSION_KEY_URL_PREFIX)

from apps.lib import clear_flash, get_datetimenow, is_allowed_file

from apps.app_modeling_web.config import (
    APP_NAME, ALLOWED_EXTENSIONS,  
    API_KEY_CROSS_VAL_NUM, API_KEY_DATADIR, API_KEY_DATA_COLUMNS, 
    API_KEY_MODEL_PARAMETER, API_KEY_RANDOM_STATE,
    API_KEY_RESULT_CV_SCORE, API_KEY_RESULT_VALIDATION_SET_SCORE, 
    API_KEY_RESULT_BEST_PARAM, API_KEY_RESULT_TEST_SCORE, 
    API_KEY_RESULT_MSE, API_KEY_RESULT_RMSE, API_KEY_RESULT_RMSE_MEAN, 
    API_KEY_RESULT_MEAN_FIT_TIME, API_KEY_RESULT_STD_FIT_TIME,
    API_KEY_RESULT_MEAN_SCORE_TIME, API_KEY_RESULT_STD_SCORE_TIME, 
    API_KEY_RESULT_PARAMS, API_KEY_RESULT_MEAN_TEST_SCORE, 
    API_KEY_RESULT_STD_TEST_SCORE, API_KEY_RESULT_RANK_TEST_SCORE, 
    FORM_KEY_ALPHA, FORM_KEY_CV, FORM_KEY_DEG,
    FORM_KEY_UPLOAD_DATADIR, FORM_KEY_UPLOAD_FILES, 
    FORM_KEY_DL_FILEPATH, FORM_KEY_MODEL_TYPE,
    FORM_KEY_X_VARIABLES, FORM_KEY_Y_VARIABLE,
    MODEL_TYPE_LIST,
    MODEL_PARAM_ALPHA, MODEL_PARAM_ALPHA_LIST, MODEL_PARAM_CV, 
    MODEL_PARAM_DEG, MODEL_PARAM_DEG_LIST, MODEL_PARAM_FIT_INTERCEPT, 
    MODEL_PARAM_MAXITER, MODEL_PARAM_MODEL_TYPE, MODEL_PARAM_X_COLUMNS, 
    MODEL_PARAM_Y_COLUMN,
    PREFIX_API, URL_INDEX_API, URL_INDEX_VIEW, VIEW_PREFIX,
    SESSION_KEY_FLASH, SESSION_KEY_DATADIR, 
    SESSION_KEY_DATA_COLUMNS, SESSION_KEY_INPUT_FILE, 
    SESSION_KEY_RESULT_FILE)

view = Blueprint(VIEW_PREFIX, __name__, template_folder='./templates', static_folder='./static')

upload_dir = ROOTDIR + APP_NAME + '/data/'


def str2num_list_items(l, cls):
    """Convert string to num for each items in list"""
    return [cls(s) for s in l]


@view.before_app_first_request
def setup():

    if not os.path.exists(upload_dir):
        os.mkdir(upload_dir)

    filename = ROOTDIR + APP_NAME + '/' + FILE_NAME_URL_PREFIX
    with open(filename, mode="r") as f:
        session[SESSION_KEY_URL_PREFIX] = f.read()
    f.close()
    
    session[SESSION_KEY_URL_INDEX] = URL_INDEX_VIEW

@view.before_app_request
def before_request():
    clear_flash(SESSION_KEY_FLASH)
    if session.get(SESSION_KEY_DATADIR):
        datadir = session[SESSION_KEY_DATADIR]
        status = 'Set directory: %s' % datadir
        flash(status)

@view.route('/')
def index():
    return render_template(VIEW_PREFIX + '/index.html')

@view.route('/upload', methods=['GET'])
def upload():
    if request.method == 'GET':
        return render_template(VIEW_PREFIX + '/upload.html')

@view.route('/upload/dir', methods=['POST'])
def upload_directory():
    if request.method == 'POST':

        datadir = request.form.get(FORM_KEY_UPLOAD_DATADIR)
        
        if datadir:
            datadir = upload_dir + datadir
            session[SESSION_KEY_DATADIR] = os.path.join(datadir, '') 
            if not os.path.exists(datadir):
                os.mkdir(datadir)
                status = 'The directory(%s) was created successfuly.' % datadir
            else:
                status = 'The dirname(%s) already exists.' % datadir
            flash(status)

        return render_template(VIEW_PREFIX + '/upload.html')

@view.route('/upload/dir/data', methods=['POST'])
def upload_file():
    if request.method == 'POST':

        if not session.get(SESSION_KEY_DATADIR):
            datadir = upload_dir + get_datetimenow() + '/'
        else:
            datadir = session[SESSION_KEY_DATADIR]

        if FORM_KEY_UPLOAD_FILES not in request.files:
            flash('No file part')
            return redirect(URL_INDEX_VIEW)

        file = request.files.getlist(FORM_KEY_UPLOAD_FILES)[0]

        if file.filename == '':
            flash('No selected file') 
            return redirect(URL_INDEX_VIEW)
        
        if file and is_allowed_file(file.filename, allowed_file_ext='csv'):
            filename = secure_filename(file.filename)
            filepath = os.path.join(datadir, filename)
            file.save(filepath)
            session[SESSION_KEY_INPUT_FILE] = filepath

            r = requests.get(
                session[SESSION_KEY_URL_PREFIX] + URL_INDEX_API + '/data/columns',
                json={
                    API_KEY_DATADIR: session[SESSION_KEY_INPUT_FILE]
                }
            )
            res = r.json()
            session[SESSION_KEY_DATA_COLUMNS] = res[API_KEY_DATA_COLUMNS]

        flash('File upload has successed.')
        return render_template(VIEW_PREFIX + '/upload.html')

@view.route('/validation/menu', methods=['GET', 'POST'])
def menu_validation():
    if request.method == 'GET':
        return render_template(
            VIEW_PREFIX + '/modeling_validation.html', 
            columns=session[SESSION_KEY_DATA_COLUMNS],
            model_types=MODEL_TYPE_LIST,
            alpha_list=MODEL_PARAM_ALPHA_LIST,
            deg_list=MODEL_PARAM_DEG_LIST)

@view.route('/validation/model', methods=['POST'])
def modeling_validation():

    if request.method == 'POST':
        r = requests.post(
            session[SESSION_KEY_URL_PREFIX] + URL_INDEX_API + '/modeling/validation',
            json={
                API_KEY_MODEL_PARAMETER: {
                    MODEL_PARAM_X_COLUMNS: request.form.getlist(FORM_KEY_X_VARIABLES),
                    MODEL_PARAM_Y_COLUMN: request.form.getlist(FORM_KEY_Y_VARIABLE),
                    MODEL_PARAM_CV: int(request.form.get(FORM_KEY_CV)),
                    MODEL_PARAM_ALPHA: str2num_list_items(request.form.getlist(FORM_KEY_ALPHA), float),
                    MODEL_PARAM_DEG: str2num_list_items(request.form.getlist(FORM_KEY_DEG), int),
                    MODEL_PARAM_FIT_INTERCEPT: False, #FIXME
                    MODEL_PARAM_MAXITER: 10000, #FIXME
                    MODEL_PARAM_MODEL_TYPE: request.form.get(FORM_KEY_MODEL_TYPE)
                },
                API_KEY_RANDOM_STATE: random.randint(1, 100),
                API_KEY_DATADIR: session[SESSION_KEY_INPUT_FILE]
            })
        res = r.json()
        return render_template(
            VIEW_PREFIX + '/result_validation.html', 
            best_param=res[API_KEY_RESULT_BEST_PARAM],
            cv_score=res[API_KEY_RESULT_CV_SCORE],
            mean_fit_time=res[API_KEY_RESULT_MEAN_FIT_TIME],
            mean_test_score=res[API_KEY_RESULT_MEAN_TEST_SCORE],
            std_fit_time=res[API_KEY_RESULT_STD_FIT_TIME],
            std_test_score=res[API_KEY_RESULT_STD_TEST_SCORE],
            std_score_time=res[API_KEY_RESULT_STD_SCORE_TIME],
            val_score=res[API_KEY_RESULT_VALIDATION_SET_SCORE],
        )

@view.route('/test/menu', methods=['GET'])
def menu_test():
    if request.method == 'GET':
        return render_template(
            VIEW_PREFIX + '/modeling_test.html', 
            columns=session[SESSION_KEY_DATA_COLUMNS],
            model_types=MODEL_TYPE_LIST,
            alpha_list=MODEL_PARAM_ALPHA_LIST,
            deg_list=MODEL_PARAM_DEG_LIST)
            
@view.route('/test/model', methods=['POST'])
def modeling_test():

    if request.method == 'POST':
        r = requests.post(
            session[SESSION_KEY_URL_PREFIX] + URL_INDEX_API + '/modeling/test',
            json={
                API_KEY_MODEL_PARAMETER: {
                    MODEL_PARAM_X_COLUMNS: request.form.getlist(FORM_KEY_X_VARIABLES),
                    MODEL_PARAM_Y_COLUMN: request.form.getlist(FORM_KEY_Y_VARIABLE),
                    MODEL_PARAM_CV: int(request.form.get(FORM_KEY_CV)),
                    MODEL_PARAM_ALPHA: float(request.form.get(FORM_KEY_ALPHA)),
                    MODEL_PARAM_DEG: int(request.form.get(FORM_KEY_DEG)),
                    MODEL_PARAM_FIT_INTERCEPT: False, #FIXME
                    MODEL_PARAM_MAXITER: 10000, #FIXME
                    MODEL_PARAM_MODEL_TYPE: request.form.get(FORM_KEY_MODEL_TYPE)
                },
                API_KEY_RANDOM_STATE: random.randint(1, 100),
                API_KEY_DATADIR: session[SESSION_KEY_INPUT_FILE]
            })
        res = r.json()

        return res