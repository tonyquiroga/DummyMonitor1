from modelop_sdk.dashboard.base_dashboard_monitors import DefaultHeatmapDashboardMonitor, DefaultRootDashboardMonitor
import modelop.monitors.stability as stability
import modelop.monitors.performance as performance
import modelop.schema.infer as infer
import modelop.monitors.bias as bias
import modelop.monitors.drift as drift
# Volumetrics
import modelop.monitors.volumetrics as volumetrics

import modelop_sdk.utils.logging as logger

# Actual ROI
import json


class MOCStabilityModule(DefaultHeatmapDashboardMonitor):
    """
    MOCStabilityModule
    Source: https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/stability_analysis
    """
    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}
    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")
        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)
        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )
        self.LOG.info("--- INIT MONITORING_PARAMETERS ")
        self.LOG.info("predictors: %s", MONITORING_PARAMETERS["predictors"])
        self.LOG.info("feature_dataclass: %s", MONITORING_PARAMETERS["feature_dataclass"])
        self.LOG.info("special_values: %s", MONITORING_PARAMETERS["special_values"])
        self.LOG.info("score_column: %s", MONITORING_PARAMETERS["score_column"])
        self.LOG.info("label_column: %s", MONITORING_PARAMETERS["label_column"])
        self.LOG.info("weight_column: %s", MONITORING_PARAMETERS["weight_column"])
        self.LOG.info(
            "protected_classes: %s", str(MONITORING_PARAMETERS["protected_classes"])
        )
        self.LOG.info("--- END INIT MONITORING_PARAMETERS ")

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:

        if kwargs.get("baseline") is None:
            self.LOG.error("Required baseline input not found")
            raise ValueError("Required baseline input not found")

        baseline = kwargs.get("baseline")

        if baseline.empty:
            self.LOG.error("Required baseline input can't be empty")
            raise ValueError("Required baseline input can't be empty")

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Required comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        score_column = MONITORING_PARAMETERS["score_column"]
        predictors = MONITORING_PARAMETERS["predictors"]

        # Initialize StabilityMonitor
        stability_monitor = stability.StabilityMonitor(
            df_baseline=baseline,
            df_sample=comparator,
            predictors=predictors,
            feature_dataclass=MONITORING_PARAMETERS["feature_dataclass"],
            special_values=MONITORING_PARAMETERS["special_values"],
            score_column=score_column,
            label_column=MONITORING_PARAMETERS["label_column"],
            weight_column=MONITORING_PARAMETERS["weight_column"],
        )

        # Set default n_groups for each predictor and score
        n_groups = {}
        for feature in MONITORING_PARAMETERS["numerical_columns"] + [
            MONITORING_PARAMETERS["score_column"]
        ]:
            # If a feature has more than 1 unique value, set n_groups to 2; else set to 1
            feature_has_distinct_values = int(
                (min(baseline[feature]) != max(baseline[feature]))
            )
            n_groups[feature] = 1 + feature_has_distinct_values

        # Compute stability metrics
        stability_metrics = stability_monitor.compute_stability_indices(
            n_groups=n_groups, group_cuts={}
        )

        stability_index_array = []
        for key, value in stability_metrics["values"].items():
            if stability_metrics["values"][key].get("stability_index") is not None:
                value = stability_metrics["values"][key]["stability_index"]
                if isinstance(value, (int, float, complex)):
                    stability_index_array.append(value)

        if len(stability_index_array) == 0:
            self.LOG.error(
                f" No numerics values found at the 'values' data drift generated results field {str(stability_index_array)} ")
            raise ValueError(
                f" No numerics values found at the 'values' data drift generated results field {str(stability_index_array)} ")

        characteristic_stability_max_stability_index = {
            "characteristic_stability_max_stability_index": max(stability_index_array)}

        return characteristic_stability_max_stability_index


class MOCPerformanceMonitor(DefaultHeatmapDashboardMonitor):
    """
    MOCPerformanceMonitor
    Source - https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/performance_classification
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}

    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")
        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)
        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )

        self.LOG.info("--- INIT MONITORING_PARAMETERS ")
        self.LOG.info("predictors: %s", MONITORING_PARAMETERS["predictors"])
        self.LOG.info("feature_dataclass: %s", MONITORING_PARAMETERS["feature_dataclass"])
        self.LOG.info("special_values: %s", MONITORING_PARAMETERS["special_values"])
        self.LOG.info("score_column: %s", MONITORING_PARAMETERS["score_column"])
        self.LOG.info("label_column: %s", MONITORING_PARAMETERS["label_column"])
        self.LOG.info("weight_column: %s", MONITORING_PARAMETERS["weight_column"])
        self.LOG.info(
            "protected_classes: %s", str(MONITORING_PARAMETERS["protected_classes"])
        )
        self.LOG.info("--- END INIT MONITORING_PARAMETERS ")

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:
        """
        Monitor result
        {
          "test_name": "Classification Metrics",
          "test_category": "performance",
          "test_type": "classification_metrics",
          "test_id": "performance_classification_metrics",
          "values": {
            "accuracy": 0.665,
            "precision": 0.4516,
            "recall": 0.7241,
            "f1_score": 0.5563, <--- value used for evaluation
            "auc": 0.6825,
            "confusion_matrix": [
              {
                "0": 0.455,
                "1": 0.255
              },
              {
                "0": 0.08,
                "1": 0.21
              }
            ]
          }
        }
        """
        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        # Initialize ModelEvaluator
        model_evaluator = performance.ModelEvaluator(
            dataframe=comparator,
            score_column=MONITORING_PARAMETERS["score_column"],
            label_column=MONITORING_PARAMETERS["label_column"]
        )

        # Compute classification metrics
        classification_metrics = model_evaluator.evaluate_performance(
            pre_defined_metrics="classification_metrics"
        )

        # Generating one output for evaluation
        raw_values_for_evaluation = {"statistical_performance_auc": classification_metrics["values"]["auc"]}

        return raw_values_for_evaluation


class MOCBiasMonitor(DefaultHeatmapDashboardMonitor):
    """
    MOCBiasMonitor
    Source: https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/bias_disparity

    Output:
    {
      "test_name": "Aequitas Bias",
      "test_category": "bias",
      "test_type": "bias",
      "protected_class": "gender",
      "test_id": "bias_bias_gender",
      "reference_group": "male",
      "thresholds": {
        "min": 0.8,
        "max": 1.25
      },
      "values": [
        {
          "attribute_name": "gender",
          "attribute_value": "female",
          "ppr_disparity": 0.5,
          "pprev_disparity": 0.8889,
          "precision_disparity": 1.36,
          "fdr_disparity": 0.7568,
          "for_disparity": 1.6098,
          "fpr_disparity": 0.7648,
          "fnr_disparity": 1.32,
          "tpr_disparity": 0.8976,
          "tnr_disparity": 1.15,
          "npv_disparity": 0.9159
        },
        {
          "attribute_name": "gender",
          "attribute_value": "male",
          "ppr_disparity": 1,
          "pprev_disparity": 1,
          "precision_disparity": 1,
          "fdr_disparity": 1,
          "for_disparity": 1,
          "fpr_disparity": 1,
          "fnr_disparity": 1,
          "tpr_disparity": 1,
          "tnr_disparity": 1,
          "npv_disparity": 1
        }
      ]
    }
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}

    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")
        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)
        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )

        self.LOG.info("--- INIT MONITORING_PARAMETERS ")
        self.LOG.info("predictors: %s", MONITORING_PARAMETERS["predictors"])
        self.LOG.info("feature_dataclass: %s", MONITORING_PARAMETERS["feature_dataclass"])
        self.LOG.info("special_values: %s", MONITORING_PARAMETERS["special_values"])
        self.LOG.info("score_column: %s", MONITORING_PARAMETERS["score_column"])
        self.LOG.info("label_column: %s", MONITORING_PARAMETERS["label_column"])
        self.LOG.info("weight_column: %s", MONITORING_PARAMETERS["weight_column"])
        self.LOG.info(
            "protected_classes: %s", str(MONITORING_PARAMETERS["protected_classes"])
        )
        self.LOG.info("--- END INIT MONITORING_PARAMETERS ")

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:
        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        for protected_class in MONITORING_PARAMETERS["protected_classes"]:
            # Initialize BiasMonitor
            bias_monitor = bias.BiasMonitor(
                dataframe=comparator,
                score_column=MONITORING_PARAMETERS["score_column"],
                label_column=MONITORING_PARAMETERS["label_column"],
                protected_class=protected_class,
                reference_group=None,
            )

            # Compute aequitas_bias (disparity) metrics
            bias_metrics = bias_monitor.compute_bias_metrics(
                pre_defined_test="aequitas_bias", thresholds={"min": 0.8, "max": 1.25}
            )

            result = {}
            result["bias"] = []

            # Add BiasMonitor Vanilla output
            result["bias"].append(bias_metrics)

            # Top-level metrics
            for group_dict in bias_metrics["values"]:
                result.update(
                    {
                        str(
                            protected_class
                            + "_"
                            + group_dict["attribute_value"]
                            + "_statistical_parity"
                        ): group_dict["ppr_disparity"],
                        str(
                            protected_class
                            + "_"
                            + group_dict["attribute_value"]
                            + "_impact_parity"
                        ): group_dict["pprev_disparity"],
                    }
                )

            # Compute aequitas_group (Group) metrics
            group_metrics = bias_monitor.compute_group_metrics(
                pre_defined_test="aequitas_group",
            )

            # Add BiasMonitor Vanilla output
            result["bias"].append(group_metrics)

            ppr_disparity_array = []
            for array_entry in result["bias"]:
                for iterated_dict in array_entry["values"]:
                    if iterated_dict.get("ppr_disparity") is not None:
                        ppr_disparity = iterated_dict["ppr_disparity"]
                        if isinstance(ppr_disparity, (int, float, complex)):
                            ppr_disparity_array.append(ppr_disparity)

            if len(ppr_disparity_array) == 0:
                self.LOG.error(
                    f" No numerics values found at the 'values' bias disparity generated results field {str(ppr_disparity_array)} ")
                raise ValueError(
                    f" No numerics values found at the 'values' bias disparity generated results field {str(ppr_disparity_array)} ")

            ethical_fairness_max_min_ppr_disparity = {"ethical_fairness_max_ppr_disparity": max(ppr_disparity_array),
                                                      "ethical_fairness_min_ppr_disparity": min(ppr_disparity_array)}

            self.LOG.info(f"ethical_fairness_max_min_ppr_disparity {str(ethical_fairness_max_min_ppr_disparity)}")

        return ethical_fairness_max_min_ppr_disparity


class MOCDataDriftKSModule(DefaultHeatmapDashboardMonitor):
    """
    MOCDataDriftComprehensiveModule
    Source: https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/data_drift_kolmogorov_smirnov
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}

    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        """
        A function to extract input schema from job JSON.
          Args:
              job_json (str): job JSON in a string format.
        """
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")

        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)

        self.LOG.info("Input schema definition: %s", input_schema_definition)

        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )

        self.LOG.info("numerical_columns: %s", MONITORING_PARAMETERS["numerical_columns"])
        self.LOG.info("categorical_columns: %s", MONITORING_PARAMETERS["categorical_columns"])

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:

        if kwargs.get("baseline") is None:
            self.LOG.error("Required baseline input not found")
            raise ValueError("Required baseline input not found")

        baseline = kwargs.get("baseline")

        if baseline.empty:
            self.LOG.error("Required baseline input can't be empty")
            raise ValueError("Required baseline input can't be empty")

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        numerical_columns = MONITORING_PARAMETERS["numerical_columns"]
        categorical_columns = MONITORING_PARAMETERS["categorical_columns"]

        # Initialize DriftDetector
        drift_detector = drift.DriftDetector(
            df_baseline=baseline,
            df_sample=comparator,
            numerical_columns=numerical_columns,
            categorical_columns=categorical_columns,
        )

        # Compute drift metrics
        drift_metrics = drift_detector.calculate_drift(
            pre_defined_test="Kolmogorov-Smirnov"
        )

        if drift_metrics.get("values") is None:
            self.LOG.error("Data Drift KS did not generate p_values ")
            raise ValueError("Data Drift KS did not generate p_values ")

        p_value_array = []
        for key, value in drift_metrics["values"].items():
            if isinstance(value, (int, float, complex)):
                p_value_array.append(value)

        if len(p_value_array) == 0:
            self.LOG.error(
                f" No numerics values found at the 'values' data drift generated results field {str(drift_metrics)} ")
            raise ValueError(
                f" No numerics values found at the 'values' data drift generated results field {str(drift_metrics)} ")

        data_drift_kr_result = {"max_p_value": max(p_value_array)}
        self.LOG.info(f" data_drift_ks_result {str(data_drift_kr_result)}")
        return data_drift_kr_result


class MOCVolumetricsIdentifierComparisonModule(DefaultHeatmapDashboardMonitor):
    """
    MOCVolumetricsIdentifierComparison
    Source: https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/volumetrics_identifier_comparison
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}

    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        """
        A function to extract input schema from job JSON.
          Args:
              job_json (str): job JSON in a string format.
        """
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")

        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)

        self.LOG.info("Input schema definition: %s", input_schema_definition)

        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:

        if kwargs.get("baseline") is None:
            self.LOG.error("Required baseline input not found")
            raise ValueError("Required baseline input not found")

        baseline = kwargs.get("baseline")

        if baseline.empty:
            self.LOG.error("Required baseline input can't be empty")
            raise ValueError("Required baseline input can't be empty")

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        # Get identifier_columns from MONITORING_PARAMETERS
        identifier_columns = MONITORING_PARAMETERS["identifier_columns"]

        # Initialize Volumetric monitor with 1st input DataFrame
        volumetric_monitor = volumetrics.VolumetricMonitor(baseline)

        # Compare DataFrames on identifier_columns
        identifiers_comparison = volumetric_monitor.identifier_comparison(
            comparator, identifier_columns
        )

        self.LOG.info(f"output_integrity_identifiers_match:{identifiers_comparison['values']['identifiers_match']}")

        return {"output_integrity_identifiers_match": identifiers_comparison["values"]["identifiers_match"]}


class MOCConceptDriftKSModule(DefaultHeatmapDashboardMonitor):
    """
    MOCConceptDriftKSModule
    Source: https://github.com/modelop/moc_monitors/tree/main/src/modelop/ootb_monitors/concept_drift_kolmogorov_smirnov
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JOB_JSON = {}

    MONITORING_PARAMETERS = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_raw_values_for_eval: dict = None):
        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_raw_values_for_eval = default_raw_values_for_eval

    def init(self, job_json) -> dict:
        """
        A function to extract input schema from job JSON.
          Args:
              job_json (str): job JSON in a string format.
        """
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")

        # Extract input schema from job JSON
        input_schema_definition = infer.extract_input_schema(job_json)

        self.LOG.info("Input schema definition: %s", input_schema_definition)

        # Get monitoring parameters from schema
        global MONITORING_PARAMETERS
        MONITORING_PARAMETERS = infer.set_monitoring_parameters(
            schema_json=input_schema_definition, check_schema=True
        )

        self.LOG.info("numerical_columns: %s", MONITORING_PARAMETERS["numerical_columns"])
        self.LOG.info("categorical_columns: %s", MONITORING_PARAMETERS["categorical_columns"])
        self.LOG.info("target_column: %s", MONITORING_PARAMETERS["score_column"])
        self.LOG.info("output_type: %s", MONITORING_PARAMETERS["output_type"])

    def get_monitor_identifier(self) -> str:
        return self.monitor_identifier

    def get_description(self) -> str:
        return self.description

    def is_heatmap_monitor(self) -> bool:
        return True

    def get_default_values_for_evaluation(self) -> dict:
        return self.default_raw_values_for_eval

    def execute_monitor(self, **kwargs) -> dict:

        if kwargs.get("baseline") is None:
            self.LOG.error("Required baseline input not found")
            raise ValueError("Required baseline input not found")

        baseline = kwargs.get("baseline")

        if baseline.empty:
            self.LOG.error("Required baseline input can't be empty")
            raise ValueError("Required baseline input can't be empty")

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("Comparator input can't be empty")
            raise ValueError("Comparator input can't be empty")

        target_column = MONITORING_PARAMETERS["score_column"]

        # Initialize DriftDetector
        concept_drift_detector = drift.ConceptDriftDetector(
            df_baseline=baseline,
            df_sample=comparator,
            target_column=target_column,
            output_type=MONITORING_PARAMETERS["output_type"],
        )

        # Compute concept drift metrics
        concept_drift_metrics = concept_drift_detector.calculate_concept_drift(
            pre_defined_test="Kolmogorov-Smirnov"
        )

        p_value_array = []
        for key, value in concept_drift_metrics["values"].items():
            if isinstance(value, (int, float, complex)):
                p_value_array.append(value)

        if len(p_value_array) == 0:
            self.LOG.error(
                f" No numerics values found at the 'values' concept drift generated results field {str(concept_drift_metrics)} ")
            raise ValueError(
                f" No numerics values found at the 'values' concept drift generated results field {str(concept_drift_metrics)} ")

        concept_drift_max_p_value = {"concept_drift_max_p_value": max(p_value_array)}

        self.LOG.debug(f"{concept_drift_max_p_value}")
        return concept_drift_max_p_value


"""
------ ROOT Monitors
"""


class MOCVolumetricsCountModule(DefaultRootDashboardMonitor):
    """
    MOCVolumetricsCountModule that performs Volumetrics counts
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JSON_JOB = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_execution_value: any = None):

        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_execution_value = default_execution_value

    def init(self, input_dict: dict = None) -> dict:
        self.JSON_JOB = input_dict
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")

    def execute_monitor(self, **kwargs) -> dict:

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("comparator baseline input can't be empty")
            raise ValueError("comparator baseline input can't be empty")

        # Initialize Volumetric monitor with 1st input DataFrame
        volumetric_monitor = volumetrics.VolumetricMonitor(comparator)
        count_results = volumetric_monitor.count()

        result = count_results["values"]["record_count"]

        return result


class MOCActualROIModule(DefaultRootDashboardMonitor):
    """
    MOCActualROIModule that generates the actual ROI
    """

    """
    Module version
    Major.Minor.Patch
    """
    VERSION = "0.1.0"
    LOG = logger.configure_logger()

    JSON_JOB = {}

    def __init__(self, monitor_identifier: str = None, description: str = None,
                 default_execution_value: any = None):

        self.monitor_identifier = monitor_identifier
        self.description = description
        self.default_execution_value = default_execution_value

    def init(self, input_dict: dict = None) -> dict:
        self.JSON_JOB = input_dict
        self.LOG.info(f"Init from {self.get_monitor_identifier()} - version ({self.VERSION})")

        """
          A function to set model-specific global variables used in ROI computations.
          """

        with open("modelop_parameters.json", "r") as parameters_file:
            modelop_parameters = json.load(parameters_file)

        ROI_parameters = modelop_parameters["monitoring"]["business_value"]["ROI"]
        self.LOG.info("ROI parameters: %s", ROI_parameters)

        global amount_field, label_field, score_field
        global cost_multipliers
        global positive_class_label

        amount_field = ROI_parameters["amount_field"]  # Column containing transaction amount
        score_field = ROI_parameters["score_field"]  # Column containing model prediction
        label_field = ROI_parameters["label_field"]  # Column containing ground_truth

        # ROI cost multipliers for each classification case
        cost_multipliers = ROI_parameters["cost_multipliers"]

        # Read and set label of positive class
        try:
            positive_class_label = modelop_parameters["monitoring"]["performance"]["positive_class_label"]
            self.LOG.info("Label of Positive Class: %s", positive_class_label)
        except KeyError:
            raise KeyError("model parameters should define label of positive class!")

    def execute_monitor(self, **kwargs) -> dict:
        """
            A Function to classify records & compute actual ROI given a labeled & scored DataFrame.
            Args:
                dataframe (pd.DataFrame): Slice of Production data
            Yields:
                dict: Test Result containing actual roi metrics
            """

        if kwargs.get("comparator") is None:
            self.LOG.error("Required comparator input not found")
            raise ValueError("Required comparator input not found")

        comparator = kwargs.get("comparator")

        if comparator.empty:
            self.LOG.error("comparator baseline input can't be empty")
            raise ValueError("comparator baseline input can't be empty")

        # Classify each record in dataframe
        for idx in range(len(comparator)):
            if comparator.iloc[idx][label_field] == comparator.iloc[idx][score_field]:
                comparator["record_class"] = (
                    "TP" if comparator.iloc[idx][label_field] == positive_class_label else "TN"
                )
            elif comparator.iloc[idx][label_field] < comparator.iloc[idx][score_field]:
                comparator["record_class"] = "FP"
            else:
                comparator["record_class"] = "FN"

        # Compute actual ROI
        actual_roi = self.compute_actual_roi(comparator)
        self.LOG.info(f"compute_actual_roi: {actual_roi}")
        return actual_roi

    def compute_actual_roi(self, data) -> float:
        """
        Helper function to compute actual ROI.
        Args:
            data (pd.DataFrame): Input DataFrame containing record_class
        Returns:
            float: actual ROI
        """

        actual_roi = 0
        for idx in range(len(data)):
            actual_roi += (
                    data.iloc[idx][amount_field]
                    * cost_multipliers[data.iloc[idx]["record_class"]]
            )

        return round(actual_roi, 2)


"""
------------
Methods to override by end users.
------------
"""


def build_monitors(job_json) -> []:
    """
    Method used to define monitors to be used inside the Dashboard model.
    """

    monitors_array = []

    moc_data_drift_comprehensive = MOCDataDriftKSModule(monitor_identifier="Data Drift",
                                                        description="Data Drift",
                                                        default_raw_values_for_eval={"max_p_value": -1})

    moc_molumetrics_identifier_comparison = MOCVolumetricsIdentifierComparisonModule(
        monitor_identifier="Output Integrity",
        description="Output Integrity",
        default_raw_values_for_eval={"output_integrity_identifiers_match": None})

    moc_concept_drift_ks_module = MOCConceptDriftKSModule(monitor_identifier="Concept Drift",
                                                          description="Concept Drift",
                                                          default_raw_values_for_eval={"concept_drift_max_p_value": -1})

    moc_performance_monitor = MOCPerformanceMonitor(monitor_identifier="Performance Monitor",
                                                    description="Performance monitor",
                                                    default_raw_values_for_eval={"statistical_performance_auc": -1})

    moc_stability_monitor = MOCStabilityModule(monitor_identifier="Characteristic Stability",
                                               description="Characteristic Stability",
                                               default_raw_values_for_eval={
                                                   "characteristic_stability_max_stability_index": -1})

    moc_bias_monitor = MOCBiasMonitor(monitor_identifier="Ethical Fairness",
                                      description="Ethical Fairness",
                                      default_raw_values_for_eval={"ethical_fairness_max_ppr_disparity": -1,
                                                                   "ethical_fairness_min_ppr_disparity": -1})

    ## Root Monitors
    moc_daily_inferences_module = MOCVolumetricsCountModule(monitor_identifier="allVolumetricMonitorRecordCount",
                                                            description="MOC Volumetrics Count",
                                                            default_execution_value="N/A")

    moc_actual_roi_module = MOCActualROIModule(monitor_identifier="actualROIAllTime",
                                               description="Actual ROI calculation",
                                               default_execution_value="N/A")

    monitors_array.append(moc_molumetrics_identifier_comparison)
    monitors_array.append(moc_data_drift_comprehensive)
    monitors_array.append(moc_concept_drift_ks_module)
    monitors_array.append(moc_performance_monitor)
    monitors_array.append(moc_stability_monitor)
    monitors_array.append(moc_bias_monitor)
    # Root Monitors
    monitors_array.append(moc_daily_inferences_module)
    monitors_array.append(moc_actual_roi_module)

    return monitors_array
