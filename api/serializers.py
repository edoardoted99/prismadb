from rest_framework import serializers

from embeddings.models import Dataset, Document
from explorer.models import FeatureFamily, Interpretation, SAEFeature
from sae.models import SAERun


class DatasetSerializer(serializers.ModelSerializer):
    total_docs = serializers.SerializerMethodField()
    done_docs = serializers.SerializerMethodField()
    error_docs = serializers.SerializerMethodField()
    progress_percent = serializers.SerializerMethodField()

    class Meta:
        model = Dataset
        fields = [
            "id", "name", "description", "model_name",
            "created_at", "updated_at",
            "total_docs", "done_docs", "error_docs", "progress_percent",
        ]
        read_only_fields = ["id", "created_at", "updated_at"]

    def get_total_docs(self, obj):
        return obj.total_docs()

    def get_done_docs(self, obj):
        return obj.done_docs()

    def get_error_docs(self, obj):
        return obj.error_docs()

    def get_progress_percent(self, obj):
        return obj.progress_percent()


class DocumentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = [
            "id", "external_id", "text", "status",
            "error_message", "created_at",
        ]


class DocumentDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = Document
        fields = "__all__"


class SAERunSerializer(serializers.ModelSerializer):
    dataset_name = serializers.CharField(source="dataset.name", read_only=True)
    feature_count = serializers.SerializerMethodField()

    class Meta:
        model = SAERun
        fields = [
            "id", "dataset", "dataset_name",
            "input_dim", "expansion_factor", "k_sparsity",
            "alpha_aux", "learning_rate", "batch_size", "epochs",
            "status", "error_message", "final_loss", "training_log",
            "feature_count",
            "created_at", "updated_at",
        ]
        read_only_fields = [
            "id", "input_dim", "status", "error_message",
            "final_loss", "training_log", "created_at", "updated_at",
        ]

    def get_feature_count(self, obj):
        return obj.features.count()


class SAERunCreateSerializer(serializers.ModelSerializer):
    class Meta:
        model = SAERun
        fields = [
            "dataset", "expansion_factor", "k_sparsity",
            "alpha_aux", "learning_rate", "batch_size", "epochs",
        ]


class SAEFeatureSerializer(serializers.ModelSerializer):
    class Meta:
        model = SAEFeature
        fields = [
            "id", "feature_index", "label", "description",
            "density", "max_activation", "mean_activation",
            "variance_activation",
        ]


class SAEFeatureDetailSerializer(serializers.ModelSerializer):
    class Meta:
        model = SAEFeature
        fields = "__all__"


class InterpretationSerializer(serializers.ModelSerializer):
    class Meta:
        model = Interpretation
        fields = "__all__"


class FeatureFamilySerializer(serializers.ModelSerializer):
    parent_label = serializers.CharField(source="parent_feature.label", read_only=True)
    children_count = serializers.SerializerMethodField()

    class Meta:
        model = FeatureFamily
        fields = [
            "id", "run", "parent_feature", "parent_label",
            "children_features", "children_count",
            "iteration", "size", "family_label",
        ]

    def get_children_count(self, obj):
        return obj.children_features.count()


class InferenceRequestSerializer(serializers.Serializer):
    run_id = serializers.IntegerField()
    text = serializers.CharField()


class SearchRequestSerializer(serializers.Serializer):
    dataset_id = serializers.IntegerField()
    query = serializers.CharField()
    size = serializers.IntegerField(default=10, required=False)


class HybridSearchRequestSerializer(SearchRequestSerializer):
    bm25_weight = serializers.FloatField(default=0.3, required=False)
    knn_weight = serializers.FloatField(default=0.7, required=False)
