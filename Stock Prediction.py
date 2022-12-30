import joblib
import pandas as pd
import base64
import plost
import streamlit as st
from PIL import Image
from pandas import DataFrame, concat
from yfinance import Ticker


def construct_input(dataset, n_back=10, dropnan=True):
    """
    This function takes in a dataset and two optional parameters, n_back and dropnan. It shifts the values in the
    dataset and returns a new dataset with the shifted values. If dropnan is True, it also removes any rows with
    missing values (NaN) from the returned dataset.
    :param dataset: the input dataset
    :param n_back: determines the number of time steps to shift the data by
    :param dropnan: determines whether or not to drop rows with NaN (Not a Number) values
    :return:
    """
    data = dataset.values.astype('float32')
    columns = [x.lower() for x in dataset.columns.values]
    n_vars = 1 if type(data) is list else data.shape[1]
    df = DataFrame(data)
    cols, names = list(), list()

    # input sequence (t-n, ... t-1)
    for i in range(n_back, 0, -1):
        cols.append(df.shift(i))
        names += [f'{columns[j]}[t-{i}]' for j in range(n_vars)]

    # forecast sequence (t, t+1, ... t+n)
    cols.append(df.shift(0))
    names += [f'{columns[j]}[t]' for j in range(n_vars)]

    # put it all together
    time_shifted = concat(cols, axis=1)
    time_shifted.columns = names
    # drop rows with NaN values
    if dropnan:
        time_shifted.dropna(inplace=True)
    return time_shifted


def plot(data):
    """
    This function creates a dropdown menu for the user to select a parameter from a list of column names in a
    DataFrame, and plots a line chart of the selected parameter.
    :param data: input data to be plotted
    """
    metrics = list(data.columns.values)
    metrics.remove('Dividends')
    metrics.remove('Stock Splits')
    metrics.remove('Volume')
    parameter = st.selectbox('Click below to select an asset parameter', metrics, index=3)
    data.reset_index(inplace=True)
    colors = ['red', 'blue', 'green', 'purple']
    col_dict = dict(zip(metrics, colors))

    plost.line_chart(
        color=col_dict[parameter],
        data=data,
        x='Date',
        y=parameter,
        width=650,
        pan_zoom='minimap')

@st.cache
def get_base64_of_bin_file(bin_file):
    """
    This function takes in a parameter called bin_file, which is the file path of a binary file. It opens the file in
    binary read mode, reads the data in the file, and assigns it to the variable data. It then encodes the data as a
    base64 string and returns the decoded string.
    :param bin_file: file to convert in base64
    :return: the base64 encoding of the file
    """
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()


def set_png_as_page_bg(png_file):
    """
    This function sets a specified image as the background image for a page by encoding the image data as a base64
    string and marking down the resulting string as unsafe HTML.
    It also sets the background color for certain elements to white.
    :param png_file: the image to set as the background
    """
    bin_str = get_base64_of_bin_file(png_file)
    page_bg_img = '''
        <style>
        .appview-container {
            background-image: url("data:image/png;base64,%s");
            background-size: cover;
        }
        table {
            background-color: white;
        }
        .dataframe {
            background-color: white;
        }
        canvas {
            background-color: white;
        }
        </style>
        ''' % bin_str

    st.markdown(page_bg_img, unsafe_allow_html=True)


@st.cache
def load_data():
    """
    This function loads and filters data from a Wikipedia page, returning a DataFrame with only the rows for 'AAPL',
    'MSFT', and 'IBM'.
    :return: the resulting DataFrame
    """
    components = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks')[0]
    components = components.drop('SEC filings', axis=1).set_index('Symbol')
    mask = components.index.isin(['AAPL', 'MSFT', 'IBM'])
    return components[mask]


@st.cache
def load_quotes(asset):
    """
    This function takes in a parameter called asset and uses it to create a Ticker object from the yfinance library.
    It then calls the history method on the Ticker object to retrieve the historical data for the asset, using the
    'max' period to retrieve all available data. It then returns the data as a DataFrame. This function is used to
    retrieve historical data for a given asset.
    :param asset: asset to retreive the historical data from
    :return: the data retrieved
    """
    data = Ticker(asset)
    data = data.history(period='max')
    return data


def handle_predict(model_name, asset, data):
    """
    This function takes in three parameters: model_name, asset, and data. It loads a model from a file using the joblib.
    load function, drops certain columns from the data DataFrame, reorders the columns, shifts the values in the data
    using the construct_input function, and finally uses the loaded model to predict a value using the shifted data.
    It returns the predicted value.
    :param model_name: name of the model that performs the prediction
    :param asset: asset to predict
    :param data: data of the asset to predict
    :return: the predicted value
    """
    model = joblib.load('models/' + asset + '_' + model_name + '.joblib')
    dataset = data.drop(columns=['Dividends', 'Stock Splits'])
    dataset = dataset[-11:]
    column_list = (dataset.columns.values.tolist())
    column_list.insert(0, column_list.pop())
    dataset = dataset.reindex(columns=column_list)
    df = construct_input(dataset, 10)

    return model.predict(df)[0]


def main():
    st.set_page_config(page_title="Stock Price Prediction App", page_icon="media/logo.png")
    set_png_as_page_bg('media/bg2.png')
    components = load_data()
    header_image = Image.open('media/bull&bear.png')
    bear = Image.open('media/bear.png').resize((150, 150))
    bull = Image.open('media/bull.png').resize((150, 150))
    st.image(header_image, use_column_width=True)

    title = st.empty()
    st.sidebar.title("Options")

    def label(symbol):
        a = components.loc[symbol]
        return symbol + ' - ' + a.Security

    st.sidebar.subheader('Select Asset')
    asset = st.sidebar.selectbox('Click below to select a new asset',
                                 components.index.sort_values(),
                                 format_func=label)

    title.title(components.loc[asset].Security)
    st.table(components.loc[asset].astype(str))

    data = load_quotes(asset).dropna()
    st.sidebar.subheader("Select Period")
    period = st.sidebar.selectbox('Click below to select a period',
                                  ['3 Days', '7 Days', '1 Month', '3 Months', '6 Months', '1 Year', '5 Years', 'Max'],
                                  index=5)
    days = {'3 Days': 3, '7 Days': 7, '1 Month': 30, '3 Months': 90, '6 Months': 180, '1 Year': 365, '5 Years': 1825,
            'Max': 0}
    st.subheader('Chart')
    plot(data[-days[period]:])

    st.sidebar.subheader('Select Model')

    dict_model_name = {'Light GBM Regressor': 'LGBMRegressor', 'XGBoost Regressor': 'XGBRegressor'}
    model_name = st.sidebar.selectbox('Click below to select the oracle',
                                      ['Light GBM Regressor ' + asset, 'Light GBM Regressor Generic',
                                       'XGBoost Regressor ' + asset, 'XGBoost Regressor Generic', ])

    init_asset = asset

    asset = "whole" if "Generic" in model_name else asset
    truncate = len('Generic') if model_name.endswith('Generic') else len(asset)
    model_name = model_name[:-truncate - 1]

    st.subheader("Prediction of " + model_name)
    button = st.button('Predict')
    if button:
        with st.spinner("Please wait..."):
            predict = handle_predict(dict_model_name[model_name], asset, data)
        st.success(f'The predicted closing price for the selected market will be {predict:.5f}')
        col1, col2, col3 = st.columns([8.5, 6, 7])

        with col1:
            st.write("")

        if predict >= data.tail(1)['Close'].values[0]:
            with col2:
                st.image(bull)
        else:
            with col2:
                st.image(bear)

        with col3:
            st.write("")

    stat_check = st.sidebar.checkbox('View statistics')
    quotes_check = st.sidebar.checkbox('View quotes')

    if stat_check:
        st.subheader('Statistics')
        st.table(data[-days[period]:].drop(columns=['Dividends', 'Stock Splits']).describe())

    if quotes_check:
        st.subheader(f'{components.loc[init_asset]["Security"]} historical data')
        st.write(data[-days[period]:].drop(columns=['Dividends', 'Stock Splits']))


if __name__ == '__main__':
    main()
