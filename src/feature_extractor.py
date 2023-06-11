from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, mutual_info_classif


class FeatureExtractor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(lowercase=True, token_pattern=r'\b[A-Za-z]+\b', norm=None)
        self.reduction_pca = PCA(n_components=5, whiten=False)
        self.feature_selection_model = SelectKBest(mutual_info_classif, k=5)

    def _tfidf_vectorizer(self, records):
        records_transformed = self.vectorizer.fit_transform(records)
        return records_transformed.toarray(), self.vectorizer.get_feature_names_out()

    def extract_features(self, records, n_components=5):
        df_records, _ = self._tfidf_vectorizer(records)
        reduction_pca = PCA(n_components=n_components, whiten=False)
        data_reduced = reduction_pca.fit_transform(df_records)
        return data_reduced

    def select_features(self, records, labels, n_components=5):
        df_records, _ = self._tfidf_vectorizer(records)
        feature_selection_model = SelectKBest(mutual_info_classif, k=n_components)
        selected_record_features = feature_selection_model.fit_transform(df_records, labels)
        return selected_record_features  # , feature_selection_model.get_feature_names_out()
