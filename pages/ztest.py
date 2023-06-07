import streamlit as st
import pandas as pd
from statistics import NormalDist
from statsmodels.stats.weightstats import ztest
from scipy.stats import shapiro


st.title('ZTEST')

with st.expander('view data'):
    df= pd.read_csv('https://raw.githubusercontent.com/ethanweed/pythonbook/main/Data/zeppo.csv')
    st.dataframe(df.transpose())

with st.expander('view statistics'):
    st.dataframe(df.describe().transpose())

st.write('## Constructing Hypothesis')
st.latex('H_{0} : \mu = \mu_{0}')
st.latex('H_{1} : \mu \neq \mu_{0}')

alpha = st.number_input('masukkkan nilai  alpha', step=0.001, min_value=0., max_value=1.)
null_mean = st.number_input('masukin nilai mu_0', step=0.001)

clicked = st.button('do the z test !!')

if clicked:
    alpha_z = NormalDist().inv_cdf(p=1-alpha/2)
    z_score, p_value = ztest(df['grades'], value=null_mean, alternative='two-sided')

    if abs(z_score) > alpha_z:
        st.latex('REJECT H_{0}')
    else:
        st.latex('CAN NOT REJECT H_{0}')
    st.write(f'titik kritis = {alpha_z}, hitung z = {z_score}, p_value = {p_value}')

st.write('## CHECk NORMALITY')

clicked_2 = st.button('do the shapiro test !!')

if clicked_2:
    result = shapiro(df['grades'])
    st.write(result)
    st.bar_chart(df['grades'])