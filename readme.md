## Problem description

In this exercise, we were asked to create graphs for different time periods before and after the start of COVID-19 vaccinations during the COVID-19 pandemic. To create the graphs, we relied on datasets obtained from GitHub and Kaggle, which contain information on cases, deaths (from COVID-19), and vaccinations for all countries.

After the collection of data, we selected the data we needed (through appropriate preprocessing), and for each country, we calculated a similarity (details of this similarity are analyzed below). Based on this, we created 2 graphs (period 01/01/2020 to 31/12/2020 - a period when vaccinations have not started yet, and period 01/01/2021 until 31/12/2022 - period when the vaccination has started).

The process and techniques followed are explained below.

## Dataset

The datasets we used are:

1. [COVID-19 vaccinations from OWID](https://github.com/owid/covid-19-data/blob/master/public/data/vaccinations/vaccinations.csv)
2. [COVID-19 deaths from OWID](https://github.com/owid/covid-19-data/blob/master/public/data/cases_deaths/full_data.csv)
3. [Population of each country in the world from Fernando Lasso](https://www.kaggle.com/datasets/fernandol/countries-of-the-world)

## Preprocessing

First and foremost, we did a featured selection in the datasets in order to keep only the useful (for analysis) data.

After collecting the data, we have created a common dataframe variable in which we have grouped all the aforementioned data by country.

After replacing the blank values with zero in the vaccination column and removing the population rows that have zero values, we have introduced a new column that counts each country's number of deaths per 100,000 residents. This action gives us a better picture of the situation of each country during the pandemic, and we can easily proceed with the comparison between countries because comparing deaths per 100,000 citizens is always a valid comparison independent of the country's population.

# [[1.png]]

## Similarity Details

To create graphs between countries, it is necessary to calculate the similarity for every possible pair of countries.

Based on the paper ["A Graph-based Methodology for Tracking COVID-19 in Time Series Datasets"](https://ieeexplore.ieee.org/document/9314516), for the similarity of the countries, we chose to implement the Gaussian similarity kernel that the paper used.

Similarity = $s_{ij}=e^{-\frac{d_{ij}^2}{2\sigma^2\}}$, where:

+ $d_{ij}^2$ the Euclidean distance between countries i and j.
+ $\sigma^2\$ the scaling parameter (so that a fair comparison can be made between two countries with different population sizes).

The similarity we implemented in our project is very similar to the previous mathematical formula.

Specifically, our similarity = $s_{ij}=e^{-\frac{|deaths_{i}-deaths_{j}|}{2}}$, where:

+ $deaths_{x}$ the number of deaths in the country x per 100,000 residents

Practically, in our implementation, we set $\sigma^2\$=1 as no scaling parameter is needed since the deaths compared are per 100,000 inhabitants. In this way, two countries can be compared independently of their size.

## Creation of graphs and clustering

Moving on, we created a new dataframe called 'similarityMatrix' (with columns country1, country2, and similarity). In this dataframe, we calculated similarity for every possible combination of 2 countries (for space and time efficiency, the pairs in the similarityMatrix are unique; for example, if there is the pair Finland in country1 and Greece in country2, the dataframe will deprive the pair Finland in country2 and Greece in country1).

Based on the similarityMatrix, Python creates a graph for each period (similarities smaller than 0.001 are considered 0 and are not included in the graph).

# [[2.png]]
# [[3.png]]

The graphs show us not only which countries (vertex) are connected to which others but also how similar they are.

In addition, this graph also shows us the clustering that has been done.

> For example, in the graph of the second period, we see that spectral clustering has divided the countries into 4 clusters (for instance, Poland and Romania belong to the same cluster).