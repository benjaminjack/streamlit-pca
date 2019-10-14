import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

st.title('Streamlit Principal Components Analysis Demo')

st.subheader('Raw data')
iris = px.data.iris().drop('species_id', axis=1)

st.write(iris)

st.subheader('Explore the original data')

xvar = st.selectbox('Select x-axis:', iris.columns[:-1])
yvar = st.selectbox('Select y-axis:', iris.columns[:-1])

st.write(px.scatter(iris, x=xvar, y=yvar, color='species'))

iris_scaled = StandardScaler().fit_transform(iris.drop('species', axis=1))
iris_pca = PCA()
iris_transformed = iris_pca.fit_transform(iris_scaled)

col_names = [f'component {i+1}' for i in range(iris_transformed.shape[1])]

iris_transformed_df = pd.DataFrame(iris_transformed, columns=col_names)
iris_transformed_df = pd.concat([iris_transformed_df, iris['species']], axis=1)

st.subheader('Transformed data')

st.write(iris_transformed_df)

st.subheader('Explore principal components')

xvar = st.selectbox('Select x-axis:', iris_transformed_df.columns[:-1])
yvar = st.selectbox('Select y-axis:', iris_transformed_df.columns[:-1])

st.write(px.scatter(iris_transformed_df, x=xvar, y=yvar, color='species'))

st.subheader('Explore loadings')

loadings = iris_pca.components_.T * np.sqrt(iris_pca.explained_variance_)

loadings_df = pd.DataFrame(loadings, columns=col_names)
loadings_df = pd.concat([loadings_df, 
                         pd.Series(iris.columns[0:4], name='var')], 
                         axis=1)

component = st.selectbox('Select component:', loadings_df.columns[0:4])

bar_chart = px.bar(loadings_df[['var', component]].sort_values(component), 
                   y='var', 
                   x=component, 
                   orientation='h',
                   range_x=[-1,1])


st.write(bar_chart)

st.write(loadings_df)


