"""
The following DataJoint pipeline implements the sequence of steps in the spike-sorting routine featured in the "spikeinterface" pipeline. Spikeinterface was developed by Alessio Buccino, Samuel Garcia, Cole Hurwitz, Jeremy Magland, and Matthias Hennig (https://github.com/SpikeInterface)
"""

from datetime import datetime

import datajoint as dj
import pandas as pd
import spikeinterface as si
from element_array_ephys import probe, readers
from element_interface.utils import find_full_path, memoized_result
from spikeinterface import exporters, extractors, sorters

from . import si_preprocessing

log = dj.logger

schema = dj.schema()

ephys = None


def activate(
    schema_name,
    *,
    ephys_module,
    create_schema=True,
    create_tables=True,
):
    """
    activate(schema_name, *, create_schema=True, create_tables=True, activated_ephys=None)
        :param schema_name: schema name on the database server to activate the `spike_sorting` schema
        :param ephys_module: the activated ephys element for which this `spike_sorting` schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    global ephys
    ephys = ephys_module
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects=ephys.__dict__,
    )
    ephys.Clustering.key_source -= PreProcessing.key_source.proj()


SI_SORTERS = [s.replace("_", ".") for s in si.sorters.sorter_dict.keys()] + [
    "consensus"
]


@schema
class PreProcessing(dj.Imported):
    """A table to handle preprocessing of each clustering task. The output will be serialized and stored as a si_recording.pkl in the output directory."""

    definition = """
    -> ephys.ClusteringTask
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@ephys-processed
        """

    @property
    def key_source(self):
        return (
            ephys.ClusteringTask * ephys.ClusteringParamSet
            & {"task_mode": "trigger"}
            & f"clustering_method in {tuple(SI_SORTERS)}"
        ) - ephys.Clustering

    def make(self, key):
        """Triggers or imports clustering analysis."""
        execution_time = datetime.utcnow()

        # Set the output directory
        clustering_method, acq_software, output_dir, params = (
            ephys.ClusteringTask * ephys.EphysRecording * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "acq_software", "clustering_output_dir", "params")

        # Get sorter method and create output directory.
        sorter_name = clustering_method.replace(".", "_")

        for required_key in (
            "SI_PREPROCESSING_METHOD",
            "SI_SORTING_PARAMS",
            "SI_POSTPROCESSING_PARAMS",
        ):
            if required_key not in params:
                raise ValueError(
                    f"{required_key} must be defined in ClusteringParamSet for SpikeInterface execution"
                )

        # Set directory to store recording file.
        if not output_dir:
            output_dir = ephys.ClusteringTask.infer_output_dir(
                key, relative=True, mkdir=True
            )
            # update clustering_output_dir
            ephys.ClusteringTask.update1(
                {**key, "clustering_output_dir": output_dir.as_posix()}
            )
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        recording_dir = output_dir / sorter_name / "recording"
        recording_dir.mkdir(parents=True, exist_ok=True)
        recording_file = recording_dir / "si_recording.pkl"

        # Create SI recording extractor object
        if acq_software == "SpikeGLX":
            spikeglx_meta_filepath = ephys.get_spikeglx_meta_filepath(key)
            spikeglx_recording = readers.spikeglx.SpikeGLX(
                spikeglx_meta_filepath.parent
            )
            spikeglx_recording.validate_file("ap")
            data_dir = spikeglx_meta_filepath.parent

            si_extractor = (
                si.extractors.neoextractors.spikeglx.SpikeGLXRecordingExtractor
            )
            stream_names, stream_ids = si.extractors.get_neo_streams(
                "spikeglx", folder_path=data_dir
            )
            si_recording: si.BaseRecording = si_extractor(
                folder_path=data_dir, stream_name=stream_names[0]
            )
        elif acq_software == "Open Ephys":
            oe_probe = ephys.get_openephys_probe_data(key)
            assert len(oe_probe.recording_info["recording_files"]) == 1
            data_dir = oe_probe.recording_info["recording_files"][0]
            si_extractor = (
                si.extractors.neoextractors.openephys.OpenEphysBinaryRecordingExtractor
            )

            stream_names, stream_ids = si.extractors.get_neo_streams(
                "openephysbinary", folder_path=data_dir
            )
            si_recording: si.BaseRecording = si_extractor(
                folder_path=data_dir, stream_name=stream_names[0]
            )
        else:
            raise NotImplementedError(
                f"SpikeInterface processing for {acq_software} not yet implemented."
            )

        # Add probe information to recording object
        electrodes_df = (
            (
                ephys.EphysRecording.Channel
                * probe.ElectrodeConfig.Electrode
                * probe.ProbeType.Electrode
                & key
            )
            .fetch(format="frame")
            .reset_index()
        )

        # Create SI probe object
        si_probe = readers.probe_geometry.to_probeinterface(
            electrodes_df[["electrode", "x_coord", "y_coord", "shank"]]
        )
        si_probe.set_device_channel_indices(electrodes_df["channel_idx"].values)
        si_recording.set_probe(probe=si_probe, in_place=True)

        # Run preprocessing and save results to output folder
        si_preproc_func = getattr(si_preprocessing, params["SI_PREPROCESSING_METHOD"])
        si_recording = si_preproc_func(si_recording)
        si_recording.dump_to_pickle(file_path=recording_file, relative_to=output_dir)

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )
        # Insert result files
        self.File.insert(
            [
                {**key, "file_name": f.relative_to(recording_dir).as_posix(), "file": f}
                for f in recording_dir.rglob("*")
                if f.is_file()
            ]
        )


@schema
class SIClustering(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> PreProcessing
    ---
    execution_time: datetime        # datetime of the start of this step
    execution_duration: float       # execution duration in hours
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@ephys-processed
        """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load recording object.
        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        clustering_method = clustering_method.replace(".", "_")
        recording_file = (
            output_dir / clustering_method / "recording" / "si_recording.pkl"
        )
        si_recording: si.BaseRecording = si.load_extractor(
            recording_file, base_folder=output_dir
        )

        sorting_params = params["SI_SORTING_PARAMS"]
        sorters_to_run = []

        if clustering_method == "consensus":
            assert isinstance(
                sorting_params, list
            ), "Consensus method requires a list of each sorter's parameters"
            for i, param_set in enumerate(sorting_params):
                sorters_to_run.append(
                    {
                        "sorter_name": param_set["sorter_name"].replace(".", "_"),
                        "sorting_output_dir": output_dir
                        / clustering_method
                        / f'{param_set["sorter_name"].replace(".", "_")}_params{i}'
                        / "spike_sorting",
                        "sorter_params": param_set["sorter_params"],
                    }
                )
        else:
            assert isinstance(
                sorting_params, dict
            ), "Single sorter methods requires a dict of sorting parameters"
            sorters_to_run = [
                {
                    "sorter_name": clustering_method,
                    "sorting_output_dir": output_dir
                    / clustering_method
                    / "spike_sorting",
                    "sorter_params": sorting_params,
                }
            ]

        sorter_objects = []
        sorter_names = []
        result_files = []
        for i, sorter in enumerate(sorters_to_run):
            # Run sorting
            @memoized_result(
                uniqueness_dict=sorter["sorter_params"],
                output_directory=sorter["sorting_output_dir"],
            )
            def _run_sorter():
                # Sorting performed in a dedicated docker environment if the sorter is not built in the spikeinterface package.
                si_sorting: si.sorters.BaseSorter = si.sorters.run_sorter(
                    sorter_name=sorter["sorter_name"],
                    recording=si_recording,
                    folder=sorter["sorting_output_dir"],
                    remove_existing_folder=True,
                    verbose=True,
                    docker_image=sorter["sorter_name"]
                    not in si.sorters.installed_sorters(),
                    **sorter["sorter_params"],
                )

                # Save sorting object
                sorting_save_path = sorter["sorting_output_dir"] / "si_sorting.pkl"
                si_sorting.dump_to_pickle(sorting_save_path, relative_to=output_dir)
                return si_sorting

            sorter_objects.append(_run_sorter())
            sorter_names.append(f"{sorter['sorter_name']}_params{i}")
            result_files.extend(
                [
                    {
                        **key,
                        "file_name": f.relative_to(
                            sorter["sorting_output_dir"]
                        ).as_posix(),
                        "file": f,
                    }
                    for f in sorter["sorting_output_dir"].rglob("*")
                    if f.is_file()
                ]
            )

        if clustering_method == "consensus":
            import spikeinterface.comparison as sc
            from spikeinterface.curation import remove_duplicated_spikes

            min_agreement = params["SI_CONSENSUS_PARAMS"].get(
                "min_agree", len(sorter_objects)
            )
            log.info(
                f"Running SI consensus requiring agreement from {min_agreement}/{len(sorter_objects)} sorters"
            )

            consensus = sc.compare_multiple_sorters(
                sorting_list=sorter_objects,
                name_list=sorter_names,
                verbose=False,
                delta_time=0.2,
                match_score=0.3,
                spiketrain_mode="union",
            )
            agreement = consensus.get_agreement_sorting(minimum_agreement_count=2)
            agreement = remove_duplicated_spikes(agreement)
            unit_labels = []
            for si_id, sorter_ids in zip(
                agreement.unit_ids, agreement.get_property("unit_ids")
            ):
                sorter_labels = [
                    sorter_objects[i].get_unit_property(
                        sorter_ids[sorter_name], "KSLabel"
                    )
                    for i, sorter_name in enumerate(sorter_names)
                ]
                label = "mua" if "mua" in sorter_labels else "good"
                unit_labels.append(label)
            agreement.set_property("KSLabel", unit_labels)

            consensus_folder = (
                output_dir / clustering_method / "spike_sorting" / "si_sorting.pkl"
            )
            agreement.dump_to_pickle(consensus_folder, relative_to=output_dir)
            result_files.extend(
                [
                    {
                        **key,
                        "file_name": f.relative_to(consensus_folder).as_posix(),
                        "file": f,
                    }
                    for f in consensus_folder.rglob("*")
                    if f.is_file()
                ]
            )

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )
        # Insert result files
        self.File.insert(result_files)


@schema
class PostProcessing(dj.Imported):
    """A processing table to handle each clustering task."""

    definition = """
    -> SIClustering
    ---
    execution_time: datetime   # datetime of the start of this step
    execution_duration: float  # execution duration in hours
    do_si_export=0: bool       # whether to export to phy
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@ephys-processed
        """

    def make(self, key):
        execution_time = datetime.utcnow()

        # Load recording & sorting object.
        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")

        recording_file = output_dir / sorter_name / "recording" / "si_recording.pkl"
        sorting_file = output_dir / sorter_name / "spike_sorting" / "si_sorting.pkl"

        si_recording: si.BaseRecording = si.load_extractor(
            recording_file, base_folder=output_dir
        )
        si_sorting: si.sorters.BaseSorter = si.load_extractor(
            sorting_file, base_folder=output_dir
        )

        postprocessing_params = params["SI_POSTPROCESSING_PARAMS"]

        job_kwargs = postprocessing_params.get(
            "job_kwargs", {"n_jobs": -1, "chunk_duration": "1s"}
        )

        analyzer_output_dir = output_dir / sorter_name / "sorting_analyzer"

        @memoized_result(
            uniqueness_dict=postprocessing_params,
            output_directory=analyzer_output_dir,
        )
        def _sorting_analyzer_compute():
            # Sorting Analyzer
            sorting_analyzer = si.create_sorting_analyzer(
                sorting=si_sorting,
                recording=si_recording,
                format="binary_folder",
                folder=analyzer_output_dir,
                sparse=True,
                overwrite=True,
            )

            # The order of extension computation is drawn from sorting_analyzer.get_computable_extensions()
            # each extension is parameterized by params specified in extensions_params dictionary (skip if not specified)
            extensions_params = postprocessing_params.get("extensions", {})
            extensions_to_compute = {
                ext_name: extensions_params[ext_name]
                for ext_name in sorting_analyzer.get_computable_extensions()
                if ext_name in extensions_params
            }

            sorting_analyzer.compute(extensions_to_compute, **job_kwargs)

        _sorting_analyzer_compute()

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
                "do_si_export": postprocessing_params.get("export_to_phy", False)
                or postprocessing_params.get("export_report", False),
            }
        )

        # Insert result files
        self.File.insert(
            [
                {
                    **key,
                    "file_name": f.relative_to(analyzer_output_dir).as_posix(),
                    "file": f,
                }
                for f in analyzer_output_dir.rglob("*")
                if f.is_file()
            ]
        )

        # Once finished, insert this `key` into ephys.Clustering
        ephys.Clustering.insert1(
            {**key, "clustering_time": datetime.utcnow()}, allow_direct_insert=True
        )


@schema
class SIExport(dj.Computed):
    """A SpikeInterface export report and to Phy"""

    definition = """
    -> PostProcessing
    ---
    execution_time: datetime
    execution_duration: float
    """

    class File(dj.Part):
        definition = """
        -> master
        file_name: varchar(255)
        ---
        file: filepath@ephys-processed
        """

    @property
    def key_source(self):
        return PostProcessing & "do_si_export = 1"

    def make(self, key):
        execution_time = datetime.utcnow()

        clustering_method, output_dir, params = (
            ephys.ClusteringTask * ephys.ClusteringParamSet & key
        ).fetch1("clustering_method", "clustering_output_dir", "params")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)
        sorter_name = clustering_method.replace(".", "_")

        postprocessing_params = params["SI_POSTPROCESSING_PARAMS"]

        job_kwargs = postprocessing_params.get(
            "job_kwargs", {"n_jobs": -1, "chunk_duration": "1s"}
        )

        analyzer_output_dir = output_dir / sorter_name / "sorting_analyzer"
        sorting_analyzer = si.load_sorting_analyzer(folder=analyzer_output_dir)

        @memoized_result(
            uniqueness_dict=postprocessing_params,
            output_directory=analyzer_output_dir / "phy",
        )
        def _export_to_phy():
            # Save to phy format
            si.exporters.export_to_phy(
                sorting_analyzer=sorting_analyzer,
                output_folder=analyzer_output_dir / "phy",
                use_relative_path=True,
                remove_if_exists=True,
                **job_kwargs,
            )

        @memoized_result(
            uniqueness_dict=postprocessing_params,
            output_directory=analyzer_output_dir / "spikeinterface_report",
        )
        def _export_report():
            # Generate spike interface report
            si.exporters.export_report(
                sorting_analyzer=sorting_analyzer,
                output_folder=analyzer_output_dir / "spikeinterface_report",
                **job_kwargs,
            )

        if postprocessing_params.get("export_report", False):
            _export_report()
        if postprocessing_params.get("export_to_phy", False):
            _export_to_phy()

        self.insert1(
            {
                **key,
                "execution_time": execution_time,
                "execution_duration": (
                    datetime.utcnow() - execution_time
                ).total_seconds()
                / 3600,
            }
        )

        # Insert result files
        for report_dirname in ("spikeinterface_report", "phy"):
            self.File.insert(
                [
                    {
                        **key,
                        "file_name": f.relative_to(analyzer_output_dir).as_posix(),
                        "file": f,
                    }
                    for f in (analyzer_output_dir / report_dirname).rglob("*")
                    if f.is_file()
                ]
            )
