    def write_all_queries_eval_results(self, experiment_dir, selected_measures, conf_level):
        """
        Writes the mean and confidence interval results for all queries in an experiment.
        """
        mean_results = {}
        ci_results = {}

        for measure in selected_measures:
            self.experiment_logger.debug("measure ", measure)
            values = [result[measure] for result in self.all_query_eval_results.values() if measure in result]
            self.experiment_logger.debug("values ", values)
            if values:
                mean_value = np.mean(values)
                std_dev = np.std(values, ddof=1)
                z = norm.ppf((1 + conf_level) / 2)
                margin_of_error = z * std_dev / np.sqrt(len(values))

                mean_results[f"mean_{measure}"] = mean_value
                ci_results[f"mean_{measure}_ub_ci"] = mean_value + margin_of_error
                ci_results[f"mean_{measure}_lb_ci"] = mean_value - margin_of_error

        all_eval_results = {**mean_results, **ci_results}
        all_eval_results_path = Path(experiment_dir) / "all_queries_eval_results.jsonl"
        with open(all_eval_results_path, "w") as file:
            json.dump(all_eval_results, file)
            file.write("\n")