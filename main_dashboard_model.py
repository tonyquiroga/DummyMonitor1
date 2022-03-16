from datetime import datetime
import custom_dashboard_monitors
import pandas as pd
import modelop_sdk.dashboard.dashboard_monitor_service as dashboard_monitor_service
import modelop_sdk.utils.logging as logger
from modelop_sdk.utils import dict_utils
import json

LOG = logger.configure_logger()

INPUT_JSON = {}
INIT_PARAM = {}


# modelop.init
def init(init_param):
    global INIT_PARAM
    INIT_PARAM = init_param
    LOG.debug(f"init function input: {str(INIT_PARAM)}")


# modelop.metrics
def metrics(baseline, comparator) -> dict:
    LOG.info("Building monitors")
    monitors_array = custom_dashboard_monitors.build_monitors(INIT_PARAM)

    if monitors_array is None or len(monitors_array) == 0:
        LOG.error("No monitors defined to be executed")
        raise ValueError("No monitors defined to be executed")
    LOG.info(f"Monitors built - total ({len(monitors_array)})")

    LOG.info("Executing monitors")
    # execute monitors
    monitor_execution_results = execute_monitors(monitors_array=monitors_array, baseline=baseline,
                                                 comparator=comparator)
    LOG.info("Monitors execution finished")

    LOG.info(f"Results for evaluation: {str(monitor_execution_results)}")
    LOG.info("Starting evaluation")

    # performing evaluation here
    evaluated_results = dashboard_monitor_service.evaluate_results(monitor_execution_results,
                                                                   "demo_dashboard_model.dmn")
    LOG.info(f"Evaluated Results: {str(evaluated_results)}")
    LOG.info("Evaluation finished")

    dashboard_result = {
        "createdDate": datetime.now().strftime('%m/%d/%Y %H:%M:%S')
    }

    # Root results for dashboard
    try:
        LOG.info("Starting root results generation")
        root_results = dashboard_monitor_service.generate_root_results(monitors_array)
        dashboard_result.update(root_results)
        LOG.info("Root results generation finished")
    except Exception as e:
        LOG.error(
            f"Root results generation finished with error - something went wrong with the dashboard root result generation, {str(e)}")
        dashboard_result.update(
            {
                "root_elements_generation_error": f"something went wrong with the dashboard root result generation"})

    # Generating heatmap
    if evaluated_results.response.ok:
        LOG.info("Generating heatmap")
        heatmap = dashboard_monitor_service.generate_heatmap_with_summary(monitors_array, evaluated_results.json)
        dashboard_result.update({"heatMap": heatmap})
        LOG.info("Heatmap generation finished")
        # Flat Root results from heatmap for further evaluation
        try:
            LOG.info("Flat Root results from heatmap for further evaluation")
            flat_heatmap = dict_utils.flatten_data(heatmap)
            dashboard_result.update(flat_heatmap)
        except Exception as e:
            LOG.error(f"Error: something went wrong with the flat heatmap generation, {str(e)}")
            dashboard_result.update(
                {
                    "flat_heatmap_generation_error": f"something went wrong with the flat heatmap generation, {str(e)}"})
    else:
        LOG.error(
            f"Heatmap generation finished with error - there was an error with the evaluation {evaluated_results.response}")
        dashboard_result.update(
            {"evaluationError": f"There was an error with the evaluation"})

    try:
        execution_details = dashboard_monitor_service.generate_execution_details(monitors_array)
        dashboard_result.update({"executionDetails": execution_details})
    except Exception as e:
        LOG.error(
            f" executionDetails field generation finished with error - there was an error with the evaluation {str(e)}")
        dashboard_result.update(
            {"executionDetails": f"field generation finished with error"})

    yield dashboard_result


def execute_monitors(**kwargs) -> dict:
    """
    Method that execute all monitors defined
    """
    monitor_execution_results = {}
    monitor_index = 1

    monitors_array = kwargs["monitors_array"]

    for monitor in monitors_array:
        # If input does not contain raw value to be evaluated nor evaluated value, we execute monitor
        try:
            LOG.info(
                f"({monitor_index}/{len(monitors_array)}) - Executing monitor '{monitor.get_monitor_identifier()}' ")
            monitor.init(INIT_PARAM)
            execution_result = monitor.execute_monitor(**kwargs)
            monitor.set_execution_result(execution_result)
            if monitor.is_evaluation_required():
                monitor_execution_results.update(execution_result)

            monitor.set_execution_status("SUCCESS")
            LOG.info(f"Monitor '{monitor.get_monitor_identifier()}' executed successfully")
        except Exception as e:
            LOG.error(
                f"Something wrong happened with '{monitor.get_monitor_identifier()}' , sending default execution values, error details: " + str(
                    e))
            monitor.set_execution_status("ERROR")
            monitor.set_error_details(f"Error executing:{monitor.get_monitor_identifier()}, error: {str(e)}")
            try:
                if monitor.is_evaluation_required():
                    monitor_execution_results.update(monitor.get_default_values_for_evaluation())
                monitor.set_execution_result(monitor.get_default_values_for_evaluation())
            except Exception as e2:
                LOG.critical(
                    f"Something went wrong while handling the exception for the current Monitor , error details: {str(e2)} ")

        monitor_index += 1

    return monitor_execution_results


# if __name__ == "__main__":
#
#     baseline_scored = pd.read_json('./sampleData/df_baseline_scored.json', lines=True)
#     comparator_scored = pd.read_json('./sampleData/df_sample_scored.json', lines=True)
#
#     # Updated job, because input schema was all in UPPER CAPS
#     jobAsString = open('./sampleData/jsonJobWithInputSchema_germanCredit_updated.json', 'r').read()
#     # jobAsString = open('./sampleData/jsonJobWithInputSchema_germanCredit_updated_bad_input_schema.json', 'r').read()
#     # jobAsString = open('sampleData/jsonJobGermanCredit_withInputSchema_error.json', 'r').read()
#     json_job_dict = {"rawJson": jobAsString}
#
#     init(json_job_dict)
#     metrics_result = metrics(baseline=baseline_scored, comparator=comparator_scored)
#     # metrics_result = metrics(baseline=None, comparator=None)
#     for result in metrics_result:
#         print(result)
#         try:
#             jsonObj = json.loads(str(result).replace("'", "\""))
#             print(json.dumps(jsonObj, indent=2))
#         except:
#             print(str(result).replace("'", "\""))
