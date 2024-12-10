"""
Python package `hsledge`: semantic evaluation for clustering results.

The package performs an evaluation of clustering results through
the semantic relationship between the significant frequent patterns
identified among the cluster items.

The method uses an internal validation technique to evaluate
the cluster rather than using distance-related metrics.
However, the algorithm requires that the data be organized in CATEGORICAL FORM.

Questions and information contact us: <aquinordga@gmail.com>
"""

import pandas as pd
import numpy as np
import math
import statistics
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

def particularize_descriptors(descriptors, particular_threshold=1.0):
    """
    Particularization of descriptors based on support.

    This function particularizes descriptors using a threshold applied
    on the carrier (support maximum - support minimum) of the feature in the
    clusters.

    Parameters
    ----------
    particular_threshold: float
        Particularization threshold.  Given the relative support,
        0.0 means that the entire range of relative support will be used,
        while 0.5 will be used half, and 1.0 only maximum support is kept.

    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the support of features in each cluster.

    Returns
    -------
    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the computed particularized support of features in each
        cluster.
    """

    for feature in descriptors.columns:
        column = np.array(descriptors[feature])

        minimum_support = np.min(column)
        maximum_support = np.max(column)

        toremove = column < minimum_support + \
            particular_threshold * (maximum_support - minimum_support)
        descriptors.loc[toremove, feature] = 0.0

    return descriptors


def semantic_descriptors(X, labels, particular_threshold=None, report_form=False):
    """
    Semantic descriptors based on feature support.
    This function computes the support of the present feature (1-itemsets
    composed by the features with value 1) of the samples in each cluster.
    Features in a cluster that do not meet the *particularization criterion*
    have their support zeroed.
    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    particular_threshold: {None, float}
        Particularization threshold.  `None` means no particularization
        strategy.
    Returns
    -------
    descriptors: array-like of shape (n_clusters, n_features)
        Matrix with the computed particularized support of features in each
        cluster.
    """

    n_clusters = max(labels) + 1

    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=[f'X{i}' for i in range(X.shape[1])])

    features = X.columns

    # 1-itemsets, for greater k we need a different algorithm
    support = X.groupby(labels).mean()

    if particular_threshold is not None:
        support = particularize_descriptors(
            support, particular_threshold=particular_threshold)
        
    if report_form is True:
        report = dict()
        for i in range(n_clusters):    
            report[i] = support.loc[i][support.loc[i] > 0].sort_values(ascending=False)
        return report
    else:
        return support

##### HammerSLEDge #####

def hsledge_score_clusters(
        X,
        labels,
        W=[.3, .1, .5, .1],
        particular_threshold=None,
        aggregation='median'):
    """
    Computes the HSLEDge score for clusters in a dataset.

    The HSLEDge score evaluates clusters based on four indicators: 
    Support (S), Length deviation (L), Exclusivity (E), and Descriptor support 
    Difference (D). These metrics are aggregated to provide an overall score for 
    each cluster, guiding the assessment of cluster quality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix where each row represents a sample and each column a feature.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample. Labels should be integers starting from 0.
    W : array-like of shape (4,), optional (default=[0.3, 0.1, 0.5, 0.1])
        Weighting factors for the SLED indicators: Support (S), Length deviation (L),
        Exclusivity (E), and Difference (D), respectively.
    particular_threshold : float or None, optional (default=None)
        Threshold for particularization of descriptors. If `None`, no threshold is applied.
    aggregation : {'harmonic', 'geometric', 'median', None}, optional (default='median')
        Aggregation method for the SLED indicators:
        - 'harmonic': Harmonic mean.
        - 'geometric': Geometric mean.
        - 'median': Median value.
        - None: Returns the S, L, E, and D scores without aggregation.

    Returns
    -------
    scores : array-like of shape (n_clusters,)
        Aggregated HSLEDge score for each cluster if `aggregation` is specified.
    score_matrix : array-like of shape (n_clusters, 4)
        Scores for S, L, E, and D for each cluster if `aggregation` is `None`.

    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.random.randint(0, 2, (100, 5))  # Binary dataset
    >>> labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    >>> scores, score_matrix = hsledge_score_clusters(X, labels, aggregation=None)
    >>> print(scores)
    >>> print(score_matrix)
    """


    n_clusters = max(labels) + 1
    descriptors = semantic_descriptors(
        X, labels, particular_threshold=particular_threshold).transpose()

    # S: Average support for descriptors (features with particularized support
    # greater than zero)
    def mean_gt_zero(x): return 0 if np.count_nonzero(
        x) == 0 else np.mean(x[x > 0])
    support_score = [mean_gt_zero(descriptors[cluster])
                     for cluster in range(n_clusters)]

    # L: Description set size deviation
    descriptor_set_size = np.array([np.count_nonzero(descriptors[cluster]) for
                                   cluster in range(n_clusters)])

    average_set_size = np.mean(descriptor_set_size[descriptor_set_size > 0])
    length_score = [0 if set_size == 0 else 1.0 /
                    (1.0 +
                     abs(set_size -
                         average_set_size)) for set_size in descriptor_set_size]

    # E: Exclusivity
    descriptor_sets = np.array([frozenset(
        descriptors.index[descriptors[cluster] > 0]) for cluster in range(n_clusters)])
    exclusive_sets = [
        descriptor_sets[cluster].difference(
            frozenset.union(
                *
                np.delete(
                    descriptor_sets,
                    cluster))) for cluster in range(n_clusters)]
    exclusive_score = [0 if len(descriptor_sets[cluster]) == 0 else len(
        exclusive_sets[cluster]) / len(descriptor_sets[cluster]) for cluster in range(n_clusters)]

    # D: Maximum ordered support difference
    ordered_support = [np.sort(descriptors[cluster])
                       for cluster in range(n_clusters)]
    diff_score = [math.sqrt(np.max(np.diff(ordered_support[cluster])))
                  for cluster in range(n_clusters)]
        
    score = pd.DataFrame.from_dict({'S': [W[0] * s for s in support_score],
                                    'L': [W[1] * l for l in length_score],
                                    'E': [W[2] * e for e in exclusive_score],
                                    'D': [W[3] * d for d in diff_score]})

    if aggregation == 'harmonic':
        score = score.transpose().apply(statistics.harmonic_mean)
    elif aggregation == 'geometric':
        score = score.transpose().apply(statistics.geometric_mean)
    elif aggregation == 'median':
        score = score.transpose().apply(statistics.median)
    else:
        assert aggregation is None

    return score


def hsledge_score(
        X,
        labels,
        W=[.3, .1, .5, .1],
        particular_threshold=None,
        aggregation='median'):
    """
    Computes the average HSLEDge score for all clusters.

    This function calculates the HSLEDge score, which evaluates the quality of clusters 
    based on four indicators: Support (S), Length deviation (L), Exclusivity (E), 
    and Descriptor support Difference (D). The scores for each cluster are computed 
    and then averaged to provide an overall measure of clustering quality.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Binary feature matrix where each row represents a sample and each column a feature.
    labels : array-like of shape (n_samples,)
        Cluster labels assigned to each sample. Labels should be integers starting from 0.
    W : array-like of shape (4,), optional (default=[0.3, 0.1, 0.5, 0.1])
        Weighting factors for the SLED indicators: Support (S), Length deviation (L),
        Exclusivity (E), and Difference (D), respectively.
    particular_threshold : float or None, optional (default=None)
        Threshold for particularization of descriptors. If `None`, no threshold is applied.
    aggregation : {'harmonic', 'geometric', 'median'}, optional (default='median')
        Aggregation method for the SLED indicators in each cluster:
        - 'harmonic': Harmonic mean of S, L, E, and D indicators.
        - 'geometric': Geometric mean of S, L, E, and D indicators.
        - 'median': Median of S, L, E, and D indicators.

    Returns
    -------
    score : float
        The average HSLEDge score across all clusters.

    Example
    -------
    >>> from sklearn.cluster import KMeans
    >>> import numpy as np
    >>> X = np.random.randint(0, 2, (100, 5))  # Binary dataset
    >>> labels = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    >>> average_score = hsledge_score(X, labels, aggregation='harmonic')
    >>> print(average_score)
    """

    assert aggregation is not None
    return np.mean(
        hsledge_score_clusters(
            X,
            labels,
            W,
            particular_threshold=particular_threshold,
            aggregation=aggregation))
            
def sledge_curve(X, labels, particular_threshold=0.0, aggregation='harmonic'):
    """
    SLEDge curve.

    This function computes the SLEDge curve.

    Parameters
    ----------
    X: array-like of shape (n_samples, n_features)
        Feature array of each sample.  All features must be binary.
    labels: array-like of shape (n_samples,)
        Cluster labels for each sample starting in 0.
    particular_threshold: {None, float}
        Particularization threshold.  `None` means no particularization
        strategy.
    aggregation: {'harmonic', 'geometric', 'median', None}
        Strategy to aggregate values of *S*, *L*, *E*, and *D*.

    Returns
    -------
    fractions: array-like of shape (>2,)
        Decreasing rate that element `i` is the fraction of clusters with
        SLEDge score >= `thresholds[i]`.  `fractions[0]` is always `1`.
    thresholds: array-like of shape (>2, )
        Increasing thresholds of the cluster SLEDge score used to compute
        `fractions`.  `thresholds[0]` is always `0` and `thresholds[-1]` is
        always `1`.
    """
    scores = sledge_score_clusters(X, labels,
                                   particular_threshold=particular_threshold,
                                   aggregation=aggregation)
    n_clusters = len(scores)

    thresholds = np.unique(scores)
    if thresholds[0] != 1:
        thresholds = np.concatenate((thresholds, [1]))
    if thresholds[len(thresholds) - 1] != 0:
        thresholds = np.concatenate(([0], thresholds))

    fractions = np.array(
        [np.count_nonzero(scores >= thr) / n_clusters for thr in thresholds])

    return fractions, thresholds


##### CDR Clustering #####

def calc_score(clusters, support):
    descriptors = [ cluster.mean(axis=0) > support for cluster in clusters ]
    exclusive_counts = []
    for c in range(len(clusters)):
        others = np.logical_or.reduce([ descriptors[k] for k in range(len(clusters)) if k != c ])
        exclusive_count = np.logical_and(descriptors[c], np.logical_not(others)).sum()
        exclusive_counts.append(exclusive_count)
    return np.mean(exclusive_counts)

def cds_clustering(X, K, support=0.8):
    clusters = [pd.DataFrame(X)]

    for k in range(K-1):
        score = []

        biggest_id = np.argmax([ x.shape[0] for x in clusters ])#[0]
        biggest = clusters[biggest_id]

        for attr in range(X.shape[1]):
            c1, c2 = biggest[biggest[attr] == 0].copy(), biggest[biggest[attr] == 1].copy()
            new_clusters = [ clusters[i] for i in range(len(clusters)) if i != biggest_id ] + [c1, c2]
            score.append(calc_score(new_clusters, support))


        best_attr = np.argmax(score)
        c1, c2 = biggest[biggest[best_attr] == 0].copy(), biggest[biggest[best_attr] == 1].copy()
        new_clusters = [ clusters[i] for i in range(len(clusters)) if i != biggest_id ] + [c1, c2]
        new_descriptors = [ np.where(cluster.mean(axis=0) > 0.8)[0] for cluster in new_clusters ]
        
        clusters = new_clusters

    labels = np.empty(X.shape[0])
    
    for i in range(len(new_clusters)):
        idx = new_clusters[i].index.values.tolist()
        for j in idx:
            labels[j] = i
    
    return labels, new_descriptors
    
def cds_report(X, K=9, support=0.8):
    clusters = [X]
    result = []
    
    while len(result) < K:
        score = []
        biggest_id = np.argmax([ x.shape[0] for x in clusters ])#[0]
        biggest = clusters[biggest_id]

        for attr in range(X.shape[1]):
            c1, c2 = biggest[biggest[:, attr] == 0, :].copy(), biggest[biggest[:, attr] == 1, :].copy()
            new_clusters = [ clusters[i] for i in range(len(clusters)) if i != biggest_id ] + [c1, c2]
            score.append(calc_score(new_clusters, support))

        best_attr = np.argmax(score)
        c1, c2 = biggest[biggest[:, best_attr] == 0, :].copy(), biggest[biggest[:, best_attr] == 1, :].copy()
        new_clusters = [ clusters[i] for i in range(len(clusters)) if i != biggest_id ] + [c1, c2]
        new_descriptors = [ np.where(cluster.mean(axis=0) > 0.8)[0] for cluster in new_clusters ]

        result.append({
            "k": len(new_clusters),
            "clusters": new_clusters,
            "score": score[best_attr],
            "descriptors": new_descriptors,
        })

        clusters = new_clusters
        
    result = pd.DataFrame(result)
    tilde_k = result.loc[result.score.idxmax()].k
    return tilde_k, result
    
 #