# app config
## file download
XLSX_MIMETYPE = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
## file upload
MAX_CONTENT_LENGTH = 16 * 1024 * 1024

## simulation
NUMBER_OF_SCENARIO = 8
ALLOWED_EXTENSIONS = {'xlsx'}

# view
VIEW_PREFIX = 'modeling_web'
URL_INDEX_VIEW = '/' + VIEW_PREFIX
APP_NAME = 'app_' + VIEW_PREFIX
FILE_NAME_URL_PREFIX = '.url_prefix'

# api
PREFIX_API = 'api_' + VIEW_PREFIX
URL_INDEX_API = '/' + PREFIX_API
SESSION_KEY_CONFIG = 'config'

API_KEY_CROSS_VAL_NUM = 'cross_val_num'
API_KEY_DATADIR = 'datadir'
API_KEY_DATA_COLUMNS = 'columns'
API_KEY_FILE_PATH = 'filepath'
API_KEY_FILE_UPDATE = 'file_update'
API_KEY_MODEL_PARAMETER = 'model_parameter'
API_KEY_RANDOM_STATE = 'random_state'

## result test
API_KEY_RESULT_MSE = 'result_mse'
API_KEY_RESULT_RMSE = 'result_rmse'
API_KEY_RESULT_RMSE_MEAN = 'result_rmse_mean'
API_KEY_RESULT_TEST_SCORE = 'result_test_score'

## result validation
API_KEY_RESULT_CV_SCORE = 'result_cv_score'
API_KEY_RESULT_VALIDATION_SET_SCORE = 'result_val_set_score'
API_KEY_RESULT_BEST_PARAM = 'result_best_param'
API_KEY_RESULT_MEAN_FIT_TIME =  'mean_fit_time'
API_KEY_RESULT_STD_FIT_TIME = 'std_fit_time'
API_KEY_RESULT_MEAN_SCORE_TIME = 'mean_score_time'
API_KEY_RESULT_STD_SCORE_TIME = 'std_score_time'
API_KEY_RESULT_PARAMS = 'params'
API_KEY_RESULT_MEAN_TEST_SCORE = 'mean_test_score'
API_KEY_RESULT_STD_TEST_SCORE = 'std_test_score'
API_KEY_RESULT_RANK_TEST_SCORE = 'rank_test_score'

## modeling validation 
GRID_PARAM_ENET_ALPHA = 'elasticnet__alpha'
GRID_PARAM_ENET_L1RATIO = 'elasticnet__l1__ratio'
GRID_PARAM_LASSO_ALPHA = 'lasso__alpha'
GRID_PARAM_POLY_DEG = 'poly__degree'
GRID_PARAM_RIDGE_ALPHA = 'ridge__alpha'

## 
MODEL_TYPE_ELNET = 'elasticnet'
MODEL_TYPE_LASSO = 'lasso'
MODEL_TYPE_RIDGE = 'ridge'
MODEL_TYPE_LIST = [MODEL_TYPE_ELNET, MODEL_TYPE_LASSO, MODEL_TYPE_RIDGE]

##
MODEL_PARAM_ALPHA = 'alpha'
MODEL_PARAM_ALPHA_LIST = [0.01, 0.1, 1, 10, 100]
MODEL_PARAM_CV = 'cv'
MODEL_PARAM_CV_LIST = list(range(10))
MODEL_PARAM_DEG = 'deg'
MODEL_PARAM_DEG_LIST = [1, 2, 3, 4]
MODEL_PARAM_FIT_INTERCEPT = 'fit_intercept'
MODEL_PARAM_L1_RATIO = 'l1_ratio'
MODEL_PARAM_MODEL_TYPE = 'model_type'
MODEL_PARAM_MAXITER = 'maxiter'
MODEL_PARAM_X_COLUMNS = 'xcolumns'
MODEL_PARAM_Y_COLUMN = 'ycolumn'

## 
PIPE_KEY_SCALAR = 'scalar'
PIPE_KEY_POLY = 'poly'

# request form key
## upload.html
FORM_KEY_ALPHA = MODEL_PARAM_ALPHA
FORM_KEY_CV = MODEL_PARAM_CV
FORM_KEY_DEG = MODEL_PARAM_DEG
FORM_KEY_MODEL_TYPE = MODEL_PARAM_MODEL_TYPE
FORM_KEY_UPLOAD_DATADIR = 'datadir'
FORM_KEY_UPLOAD_FILES = 'files'
FORM_KEY_X_VARIABLES = 'xcolumns'
FORM_KEY_Y_VARIABLE = 'ycolumn'
## download.html
FORM_KEY_DL_FILEPATH = API_KEY_FILE_PATH

# session key
SESSION_KEY_DATADIR = 'datadir'
SESSION_KEY_FLASH = '_flashes'
SESSION_KEY_INPUT_FILE = 'input_file'
SESSION_KEY_RESULT_FILE = 'result_file'
SESSION_KEY_DATA_COLUMNS = API_KEY_DATA_COLUMNS