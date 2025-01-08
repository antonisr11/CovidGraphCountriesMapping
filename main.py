import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.cluster import SpectralClustering
import warnings
warnings.filterwarnings("ignore")

def loadDataset():
    """ Loads the dataset and returns a pandas dataframe """

    # All countries #
    allCountries = pd.read_csv('data/countries of the world.csv')[['Country', 'Region', 'Population']]

    # Rename columns
    allCountries.rename(columns={'Country': 'country'}, inplace=True)
    allCountries.rename(columns={'Region': 'region'}, inplace=True)
    allCountries.rename(columns={'Population': 'population'}, inplace=True)

    # strip Country column
    allCountries['country'] = allCountries['country'].str.strip()
    allCountries['region'] = allCountries['region'].str.strip()
    allCountries['population'] = allCountries['population'].astype(int)

    # Country data #
    countryData = pd.read_csv('data/full_data.csv')[['date', 'location', 'new_cases', 'new_deaths']]

    # Rename columns
    countryData.rename(columns={'location': 'country'}, inplace=True)
    countryData.rename(columns={'new_cases': 'cases'}, inplace=True)
    countryData.rename(columns={'new_deaths': 'deaths'}, inplace=True)

    # Convert date to datetime
    countryData['date'] = pd.to_datetime(countryData['date'])

    # Vaccinations #
    vaccinations = pd.read_csv('data/vaccinations.csv')[['date', 'location', 'daily_vaccinations']]

    # Rename columns
    vaccinations.rename(columns={'location': 'country'}, inplace=True)
    vaccinations.rename(columns={'daily_vaccinations': 'vaccinations'}, inplace=True)

    # Convert date to datetime
    vaccinations['date'] = pd.to_datetime(vaccinations['date'])

    # Join the two datasets on date and country
    countryData = countryData.merge(vaccinations, on=['date', 'country'], how='left')
    countryData = countryData.merge(allCountries, on=['country'], how='left')
    del vaccinations, allCountries

    # Fill NaN vaccinations with 0
    countryData['vaccinations'] = countryData['vaccinations'].fillna(0)

    # Remove rows with NaN in population
    countryData = countryData[countryData['population'].notna()]

    # Create new column for deaths per 100,000
    countryData['deathsPer100k'] = (countryData['deaths'] / countryData['population']) * 100000

    return countryData


def plotVaccinations(countryDataTemp, title):
    """
    :param countryDataTemp: Dataframe with country data
    :param title: Title of the plot
    :return: It plots the vaccinations
    """

    countryDataTemp.plot(x='date', y='vaccinations', figsize=(20, 10), title='Vaccinations')
    plt.title(title)
    plt.show()
    return


def groupByCountry(countryDataTemp):
    """
    :param countryDataTemp: Dataframe with country data
    :return: It returns a dataframe grouped by country and region
    """
    groupedDf = countryDataTemp.groupby(['country', 'region']).agg({
        'cases': 'sum',
        'deaths': 'sum',
        'vaccinations': 'sum',
        'population': 'max'
    }).reset_index()

    # deathsPer100k column is calculated from summed deaths and population
    groupedDf['deathsPer100k'] = (groupedDf['deaths'] / groupedDf['population']) * 100000
    return groupedDf


def getSimilarityMatrix(countryDataTemp):
    """
    :param countryDataTemp: Dataframe with country data
    :return: It returns a dataframe with the similarity matrix. Similarity is calculated as:
            e^(-0.5 * abs(deathsPer100k_country_1 - deathsPer100k_country_2))
    """
    def calculateSimilarity(deathsPer100k_country_1, deathsPer100k_country_2):
        similarity = np.exp(-0.5 * abs(deathsPer100k_country_1 - deathsPer100k_country_2))
        return similarity if similarity > 0.001 else 0

    countryDataLength = len(countryDataTemp)
    similarityMatrix_Country1 = list()
    similarityMatrix_Country2 = list()
    similarityMatrix_Similarity = list()

    for i in tqdm(range(countryDataLength)):
        for j in range(i + 1, countryDataLength):
            similarityMatrix_Country1.append(countryDataTemp.iloc[i]['country'])
            similarityMatrix_Country2.append(countryDataTemp.iloc[j]['country'])
            similarityMatrix_Similarity.append(
                calculateSimilarity(countryDataTemp.iloc[i]['deathsPer100k'], countryDataTemp.iloc[j]['deathsPer100k']))

    return pd.DataFrame({'country1': similarityMatrix_Country1, 'country2': similarityMatrix_Country2,
                         'similarity': similarityMatrix_Similarity})

def plotNxGraph(countryDataTemp, title, numberOfClusters=4):
    """
    It plots the nx graph with the clusters
    :param numberOfClusters: Number of clusters to be created
    :param countryDataTemp: Dataframe with country data
    :param title: Title of the plot
    """
    def getColorForSimilarityForEdges(similarityValue):
        if similarityValue >= 0.9:
            return '#ff0000'
        elif similarityValue >= 0.8:
            return '#ff1919'
        elif similarityValue >= 0.7:
            return '#ff3232'
        elif similarityValue >= 0.6:
            return '#ff4c4c'
        elif similarityValue >= 0.5:
            return '#ff6666'
        elif similarityValue >= 0.4:
            return '#ff7f7f'
        elif similarityValue >= 0.3:
            return '#ff9999'
        elif similarityValue >= 0.2:
            return '#ffb2b2'
        elif similarityValue >= 0.1:
            return '#ffcccc'
        else:
            return '#ffe5e5'
        pass

    def getColorForCluster(clusterId):
        allColors = ['lightblue', 'lightgreen', 'yellow', 'orange', 'purple', 'pink', 'brown', 'teal', 'coral', 'red']
        return allColors[clusterId % len(allColors)]

    # Create similarity matrix
    similarityMatrix = getSimilarityMatrix(countryDataTemp)

    graph = nx.Graph()
    edges_colors = list()
    allCountries = similarityMatrix['country1'].unique()
    for _, row in similarityMatrix.iterrows():
        similarity = round(row['similarity'], 3)
        if similarity <= 0:
            continue

        graph.add_edge(row['country1'], row['country2'], weight=similarity)
        edges_colors.append(getColorForSimilarityForEdges(similarity))

    # Add nodes for countries that are not in the similarity matrix or have no similarity
    for country in allCountries:
        if country not in graph.nodes:
            graph.add_node(country)

    countryDataTemp = addClusterInCountries(countryDataTemp, graph, numberOfClusters)
    vertices_colors = list()
    for _, row in countryDataTemp.iterrows():
        vertices_colors.append(getColorForCluster(row['cluster']))

    pos = nx.arf_layout(graph)
    plt.figure()
    plt.title(title)
    nx.draw(graph, pos, with_labels=True, node_color=vertices_colors, edge_color=edges_colors, width=1, font_size=10)
    plt.axis('off')
    plt.show()
    return


def addClusterInCountries(countryDataTemp, similarityMatrix, numberOfClusters):
    """
    :param countryDataTemp: Dataframe with country data that will be clustered
    :param similarityMatrix: Dataframe with similarity matrix
    :param numberOfClusters: Number of clusters
    :return: It returns a dataframe with the cluster column
    """
    adjacency_matrix = nx.to_numpy_array(similarityMatrix)
    clusterer = SpectralClustering(n_clusters=numberOfClusters, affinity='precomputed')
    newColumnClusters = clusterer.fit_predict(adjacency_matrix)
    countryDataTemp['cluster'] = newColumnClusters
    return countryDataTemp


if __name__ == '__main__':
    # Load dataset
    print("Loading dataset...")
    countryData = loadDataset()

    # Keep only European countries (where Europe is in the region column ignoring case)
    print("Keeping only European countries...")
    countryData = countryData[countryData['region'].str.contains('europe', case=False)]

    print("Grouping by country...")
    plotVaccinations(countryData, "Total Vaccinations")

    # Period 1 has data only from 01/01/2020 to 31/12/2020
    print("Grouping by country for period 1...")
    countryData_period1 = countryData[(countryData['date'] >= '2020-01-01') & (countryData['date'] <= '2020-12-31')]
    countryData_period1 = groupByCountry(countryData_period1)

    # Make the graphs in nx
    print("Plotting graphs for period 1...")
    plotNxGraph(countryData_period1, "Countries Similarity & Clustering for period 1")

    # Period 2 has data only from 01/01/2021 to 31/12/2022
    print("Grouping by country for period 2...")
    countryData_period2 = countryData[(countryData['date'] >= '2021-01-01') & (countryData['date'] <= '2022-12-31')]
    countryData_period2 = groupByCountry(countryData_period2)

    # Make the graphs in nx
    print("Plotting graphs for period 2...")
    plotNxGraph(countryData_period2, "Countries Similarity & Clustering for period 2")