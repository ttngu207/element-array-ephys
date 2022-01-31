import datajoint as dj
import pathlib
import numpy as np
import inspect
import importlib

import make_templates
from element_interface.utils import find_root_directory, find_full_path, dict_to_uuid
from .readers import spikeglx, kilosort, openephys
from . import probe, get_logger


log = get_logger(__name__)

schema = dj.schema()

_linking_module = None


def activate(ephys_schema_name, probe_schema_name=None, *, create_schema=True,
             create_tables=True, linking_module=None):
    """
    activate(ephys_schema_name, probe_schema_name=None, *, create_schema=True, create_tables=True, linking_module=None)
        :param ephys_schema_name: schema name on the database server to activate the `ephys` element
        :param probe_schema_name: schema name on the database server to activate the `probe` element
         - may be omitted if the `probe` element is already activated
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
        :param linking_module: a module name or a module containing the
         required dependencies to activate the `ephys` element:
            Upstream tables:
                + Subject: table referenced by ProbeInsertion, typically identifying the animal undergoing a probe insertion
                + Session: table referenced by EphysRecording, typically identifying a recording session
                + SkullReference: Reference table for InsertionLocation, specifying the skull reference
                 used for probe insertion location (e.g. Bregma, Lambda)
            Functions:
                + get_ephys_root_data_dir() -> list
                    Retrieve the root data directory - e.g. containing the raw ephys recording files for all subject/sessions.
                    :return: a string for full path to the root data directory
                + get_session_directory(session_key: dict) -> str
                    Retrieve the session directory containing the recorded Neuropixels data for a given Session
                    :param session_key: a dictionary of one Session `key`
                    :return: a string for full path to the session directory
                + get_processed_root_data_dir() -> str:
                    Retrieves the root directory for all processed data to be found from or written to
                    :return: a string for full path to the root directory for processed data
    """

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global _linking_module
    _linking_module = linking_module

    probe.activate(probe_schema_name, create_schema=create_schema,
                   create_tables=create_tables)
    schema.activate(ephys_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=_linking_module.__dict__)


# -------------- Functions required by the elements-ephys  ---------------

def get_ephys_root_data_dir() -> list:
    """
    All data paths, directories in DataJoint Elements are recommended to be 
    stored as relative paths, with respect to some user-configured "root" 
    directory, which varies from machine to machine (e.g. different mounted 
    drive locations)

    get_ephys_root_data_dir() -> list
        This user-provided function retrieves the possible root data directories
         containing the ephys data for all subjects/sessions
         (e.g. acquired SpikeGLX or Open Ephys raw files,
         output files from spike sorting routines, etc.)
        :return: a string for full path to the ephys root data directory,
         or list of strings for possible root data directories
    """
    root_directories = _linking_module.get_ephys_root_data_dir()
    if isinstance(root_directories, (str, pathlib.Path)):
        root_directories = [root_directories]

    if hasattr(_linking_module, 'get_processed_root_data_dir'):
        root_directories.append(_linking_module.get_processed_root_data_dir())

    return root_directories


def get_session_directory(session_key: dict) -> str:
    """
    get_session_directory(session_key: dict) -> str
        Retrieve the session directory containing the
         recorded Neuropixels data for a given Session
        :param session_key: a dictionary of one Session `key`
        :return: a string for relative or full path to the session directory
    """
    return _linking_module.get_session_directory(session_key)


def get_processed_root_data_dir() -> str:
    """
    get_processed_root_data_dir() -> str:
        Retrieves the root directory for all processed data to be found from or written to
        :return: a string for full path to the root directory for processed data
    """

    if hasattr(_linking_module, 'get_processed_root_data_dir'):
        return _linking_module.get_processed_root_data_dir()
    else:
        return get_ephys_root_data_dir()[0]

# ----------------------------- Table declarations ----------------------


@schema
class AcquisitionSoftware(dj.Lookup):
    definition = """  # Software used for recording of neuropixels probes
    acq_software: varchar(24)    
    """
    contents = zip(['SpikeGLX', 'Open Ephys'])


@schema
class ProbeInsertion(dj.Manual):
    definition = """
    # Probe insertion chronically implanted into an animal.
    -> Subject  
    insertion_number: tinyint unsigned
    ---
    -> probe.Probe
    insertion_datetime=null: datetime
    """


@schema
class InsertionLocation(dj.Manual):
    definition = """
    # Brain Location of a given probe insertion.
    -> ProbeInsertion
    ---
    -> SkullReference
    ap_location: decimal(6, 2) # (um) anterior-posterior; ref is 0; more anterior is more positive
    ml_location: decimal(6, 2) # (um) medial axis; ref is 0 ; more right is more positive
    depth:       decimal(6, 2) # (um) manipulator depth relative to surface of the brain (0); more ventral is more negative
    theta=null:  decimal(5, 2) # (deg) - elevation - rotation about the ml-axis [0, 180] - w.r.t the z+ axis
    phi=null:    decimal(5, 2) # (deg) - azimuth - rotation about the dv-axis [0, 360] - w.r.t the x+ axis
    beta=null:   decimal(5, 2) # (deg) rotation about the shank of the probe [-180, 180] - clockwise is increasing in degree - 0 is the probe-front facing anterior
    """


@schema
class EphysRecording(dj.Imported):
    definition = """
    # Ephys recording from a probe insertion for a given session.
    -> Session
    -> ProbeInsertion      
    ---
    -> probe.ElectrodeConfig
    -> AcquisitionSoftware
    sampling_rate: float # (Hz) 
    recording_datetime: datetime # datetime of the recording from this probe
    recording_duration: float # (seconds) duration of the recording from this probe
    """

    class EphysFile(dj.Part):
        definition = """
        # Paths of files of a given EphysRecording round.
        -> master
        file_path: varchar(255)  # filepath relative to root data directory
        """

    def make(self, key):
        make_templates.EphysRecordingTemplate.make(self, key)


@schema
class LFP(dj.Imported):
    definition = """
    # Acquired local field potential (LFP) from a given Ephys recording.
    -> EphysRecording
    ---
    lfp_sampling_rate: float   # (Hz)
    lfp_time_stamps: longblob  # (s) timestamps with respect to the start of the recording (recording_timestamp)
    lfp_mean: longblob         # (uV) mean of LFP across electrodes - shape (time,)
    """

    class Electrode(dj.Part):
        definition = """
        -> master
        -> probe.ElectrodeConfig.Electrode  
        ---
        lfp: longblob               # (uV) recorded lfp at this electrode 
        """

    # Only store LFP for every 9th channel, due to high channel density,
    # close-by channels exhibit highly similar LFP
    _skip_channel_counts = 9

    def make(self, key):
        make_templates.LFPTemplate.make(self, key)


# ------------ Clustering --------------

@schema
class ClusteringMethod(dj.Lookup):
    definition = """
    # Method for clustering
    clustering_method: varchar(16)
    ---
    clustering_method_desc: varchar(1000)
    """

    contents = [('kilosort2.5', 'kilosort2.5 clustering method'),
                ('kilosort3', 'kilosort3 clustering method')]


@schema
class ClusteringParamSet(dj.Lookup):
    definition = """
    # Parameter set to be used in a clustering procedure
    paramset_idx:  smallint
    ---
    -> ClusteringMethod    
    paramset_desc: varchar(128)
    param_set_hash: uuid
    unique index (param_set_hash)
    params: longblob  # dictionary of all applicable parameters
    """

    @classmethod
    def insert_new_params(cls, clustering_method: str, paramset_desc: str,
                          params: dict, paramset_idx: int = None):
        if paramset_idx is None:
            paramset_idx = (dj.U().aggr(cls, n='max(paramset_idx)').fetch1('n') or 0) + 1

        param_dict = {'clustering_method': clustering_method,
                      'paramset_idx': paramset_idx,
                      'paramset_desc': paramset_desc,
                      'params': params,
                      'param_set_hash':  dict_to_uuid(
                          {**params, 'clustering_method': clustering_method})
                      }
        param_query = cls & {'param_set_hash': param_dict['param_set_hash']}

        if param_query:  # If the specified param-set already exists
            existing_paramset_idx = param_query.fetch1('paramset_idx')
            if existing_paramset_idx == paramset_idx:  # If the existing set has the same paramset_idx: job done
                return
            else:  # If not same name: human error, trying to add the same paramset with different name
                raise dj.DataJointError(
                    f'The specified param-set already exists'
                    f' - with paramset_idx: {existing_paramset_idx}')
        else:
            if {'paramset_idx': paramset_idx} in cls.proj():
                raise dj.DataJointError(
                    f'The specified paramset_idx {paramset_idx} already exists,'
                    f' please pick a different one.')
            cls.insert1(param_dict)


@schema
class ClusterQualityLabel(dj.Lookup):
    definition = """
    # Quality
    cluster_quality_label:  varchar(100)  # cluster quality type - e.g. 'good', 'MUA', 'noise', etc.
    ---
    cluster_quality_description:  varchar(4000)
    """
    contents = [
        ('good', 'single unit'),
        ('ok', 'probably a single unit, but could be contaminated'),
        ('mua', 'multi-unit activity'),
        ('noise', 'bad unit')
    ]


@schema
class ClusteringTask(dj.Manual):
    definition = """
    # Manual table for defining a clustering task ready to be run
    -> EphysRecording
    -> ClusteringParamSet
    ---
    clustering_output_dir='': varchar(255)  #  clustering output directory relative to the clustering root data directory
    task_mode='load': enum('load', 'trigger')  # 'load': load computed analysis results, 'trigger': trigger computation
    """

    @classmethod
    def infer_output_dir(cls, key, relative=False, mkdir=False):
        """
        Given a 'key' to an entry in this table
        Return the expected clustering_output_dir based on the following convention:
            processed_dir / session_dir / probe_{insertion_number} / {clustering_method}_{paramset_idx}
            e.g.: sub4/sess1/probe_2/kilosort2_0
        """
        processed_dir = pathlib.Path(get_processed_root_data_dir())
        session_dir = find_full_path(get_ephys_root_data_dir(),
                                  get_session_directory(key))
        root_dir = find_root_directory(get_ephys_root_data_dir(), session_dir)

        method = (ClusteringParamSet * ClusteringMethod & key).fetch1(
            'clustering_method').replace(".", "-")

        output_dir = (processed_dir
                      / session_dir.relative_to(root_dir)
                      / f'probe_{key["insertion_number"]}'
                      / f'{method}_{key["paramset_idx"]}')

        if mkdir:
            output_dir.mkdir(parents=True, exist_ok=True)
            log.info(f'{output_dir} created!')

        return output_dir.relative_to(processed_dir) if relative else output_dir

    @classmethod
    def auto_generate_entries(cls, ephys_recording_key):
        """
        Method to auto-generate ClusteringTask entries for a particular ephys recording
            Output directory is auto-generated based on the convention
             defined in `ClusteringTask.infer_output_dir()`
            Default parameter set used: paramset_idx = 0
        """
        key = {**ephys_recording_key, 'paramset_idx': 0}

        processed_dir = get_processed_root_data_dir()
        output_dir = ClusteringTask.infer_output_dir(key, relative=False, mkdir=True)

        try:
            kilosort.Kilosort(output_dir)  # check if the directory is a valid Kilosort output
        except FileNotFoundError:
            task_mode = 'trigger'
        else:
            task_mode = 'load'

        cls.insert1({
            **key,
            'clustering_output_dir': output_dir.relative_to(processed_dir).as_posix(),
            'task_mode': task_mode})


@schema
class Clustering(dj.Imported):
    """
    A processing table to handle each ClusteringTask:
    + If `task_mode == "trigger"`: trigger clustering analysis
        according to the ClusteringParamSet (e.g. launch a kilosort job)
    + If `task_mode == "load"`: verify output
    """
    definition = """
    # Clustering Procedure
    -> ClusteringTask
    ---
    clustering_time: datetime  # time of generation of this set of clustering results 
    package_version='': varchar(16)
    """

    def make(self, key):
        make_templates.ClusteringTemplate.make(self, key)


@schema
class Curation(dj.Manual):
    definition = """
    # Manual curation procedure
    -> Clustering
    curation_id: int
    ---
    curation_time: datetime             # time of generation of this set of curated clustering results 
    curation_output_dir: varchar(255)   # output directory of the curated results, relative to root data directory
    quality_control: bool               # has this clustering result undergone quality control?
    manual_curation: bool               # has manual curation been performed on this clustering result?
    curation_note='': varchar(2000)  
    """

    def create1_from_clustering_task(self, key, curation_note=''):
        """
        A function to create a new corresponding "Curation" for a particular 
        "ClusteringTask"
        """
        if key not in Clustering():
            raise ValueError(f'No corresponding entry in Clustering available'
                             f' for: {key}; do `Clustering.populate(key)`')

        task_mode, output_dir = (ClusteringTask & key).fetch1(
            'task_mode', 'clustering_output_dir')
        kilosort_dir = find_full_path(get_ephys_root_data_dir(), output_dir)

        creation_time, is_curated, is_qc = kilosort.extract_clustering_info(kilosort_dir)
        # Synthesize curation_id
        curation_id = dj.U().aggr(self & key, n='ifnull(max(curation_id)+1,1)').fetch1('n')
        self.insert1({**key, 'curation_id': curation_id,
                      'curation_time': creation_time, 
                      'curation_output_dir': output_dir,
                      'quality_control': is_qc, 
                      'manual_curation': is_curated,
                      'curation_note': curation_note})


@schema
class CuratedClustering(dj.Imported):
    definition = """
    # Clustering results of a curation.
    -> Curation    
    """

    class Unit(dj.Part):
        definition = """
        # Properties of a given unit from a round of clustering (and curation)
        -> master
        unit: int
        ---
        -> probe.ElectrodeConfig.Electrode  # electrode with highest waveform amplitude for this unit
        -> ClusterQualityLabel
        spike_count: int         # how many spikes in this recording for this unit
        spike_times: longblob    # (s) spike times of this unit, relative to the start of the EphysRecording
        spike_sites : longblob   # array of electrode associated with each spike
        spike_depths : longblob  # (um) array of depths associated with each spike, relative to the (0, 0) of the probe    
        """

    def make(self, key):
        make_templates.CuratedClusteringTemplate.make(self, key)


@schema
class WaveformSet(dj.Imported):
    definition = """
    # A set of spike waveforms for units out of a given CuratedClustering
    -> CuratedClustering
    """

    class PeakWaveform(dj.Part):
        definition = """
        # Mean waveform across spikes for a given unit at its representative electrode
        -> master
        -> CuratedClustering.Unit
        ---
        peak_electrode_waveform: longblob  # (uV) mean waveform for a given unit at its representative electrode
        """

    class Waveform(dj.Part):
        definition = """
        # Spike waveforms and their mean across spikes for the given unit
        -> master
        -> CuratedClustering.Unit
        -> probe.ElectrodeConfig.Electrode  
        --- 
        waveform_mean: longblob   # (uV) mean waveform across spikes of the given unit
        waveforms=null: longblob  # (uV) (spike x sample) waveforms of a sampling of spikes at the given electrode for the given unit
        """

    def make(self, key):
        make_templates.WaveformSetTemplate.make(self, key)
        

# ---------------- HELPER FUNCTIONS ----------------

def get_spikeglx_meta_filepath(ephys_recording_key):
    # attempt to retrieve from EphysRecording.EphysFile
    spikeglx_meta_filepath = (EphysRecording.EphysFile & ephys_recording_key
                              & 'file_path LIKE "%.ap.meta"').fetch1('file_path')

    try:
        spikeglx_meta_filepath = find_full_path(get_ephys_root_data_dir(),
                                                spikeglx_meta_filepath)
    except FileNotFoundError:
        # if not found, search in session_dir again
        if not spikeglx_meta_filepath.exists():
            session_dir = find_full_path(get_ephys_root_data_dir(), 
                                         get_session_directory(
                                             ephys_recording_key))
            inserted_probe_serial_number = (ProbeInsertion * probe.Probe
                                            & ephys_recording_key).fetch1('probe')

            spikeglx_meta_filepaths = [fp for fp in session_dir.rglob('*.ap.meta')]
            for meta_filepath in spikeglx_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    spikeglx_meta_filepath = meta_filepath
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(ephys_recording_key))

    return spikeglx_meta_filepath


def get_openephys_probe_data(ephys_recording_key):
    inserted_probe_serial_number = (ProbeInsertion * probe.Probe
                                    & ephys_recording_key).fetch1('probe')
    session_dir = find_full_path(get_ephys_root_data_dir(),
                              get_session_directory(ephys_recording_key))
    loaded_oe = openephys.OpenEphys(session_dir)
    return loaded_oe.probes[inserted_probe_serial_number]


def get_neuropixels_channel2electrode_map(ephys_recording_key, acq_software):
    if acq_software == 'SpikeGLX':
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_meta = spikeglx.SpikeGLXMeta(spikeglx_meta_filepath)
        electrode_config_key = (EphysRecording * probe.ElectrodeConfig
                                & ephys_recording_key).fetch1('KEY')

        electrode_query = (probe.ProbeType.Electrode
                           * probe.ElectrodeConfig.Electrode & electrode_config_key)

        probe_electrodes = {
            (shank, shank_col, shank_row): key
            for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                'KEY', 'shank', 'shank_col', 'shank_row'))}

        channel2electrode_map = {
            recorded_site: probe_electrodes[(shank, shank_col, shank_row)]
            for recorded_site, (shank, shank_col, shank_row, _) in enumerate(
                spikeglx_meta.shankmap['data'])}
    elif acq_software == 'Open Ephys':
        probe_dataset = get_openephys_probe_data(ephys_recording_key)

        electrode_query = (probe.ProbeType.Electrode
                           * probe.ElectrodeConfig.Electrode
                           * EphysRecording & ephys_recording_key)

        probe_electrodes = {key['electrode']: key
                            for key in electrode_query.fetch('KEY')}

        channel2electrode_map = {
            channel_idx: probe_electrodes[channel_idx]
            for channel_idx in probe_dataset.ap_meta['channels_indices']}

    return channel2electrode_map


def generate_electrode_config(probe_type: str, electrodes: list):
    """
    Generate and insert new ElectrodeConfig
    :param probe_type: probe type (e.g. neuropixels 2.0 - SS)
    :param electrodes: list of the electrode dict (keys of the probe.ProbeType.Electrode table)
    :return: a dict representing a key of the probe.ElectrodeConfig table
    """
    # compute hash for the electrode config (hash of dict of all ElectrodeConfig.Electrode)
    electrode_config_hash = dict_to_uuid({k['electrode']: k for k in electrodes})

    electrode_list = sorted([k['electrode'] for k in electrodes])
    electrode_gaps = ([-1]
                      + np.where(np.diff(electrode_list) > 1)[0].tolist()
                      + [len(electrode_list) - 1])
    electrode_config_name = '; '.join([
        f'{electrode_list[start + 1]}-{electrode_list[end]}'
        for start, end in zip(electrode_gaps[:-1], electrode_gaps[1:])])

    electrode_config_key = {'electrode_config_hash': electrode_config_hash}

    # ---- make new ElectrodeConfig if needed ----
    if not probe.ElectrodeConfig & electrode_config_key:
        probe.ElectrodeConfig.insert1({**electrode_config_key, 'probe_type': probe_type,
                                       'electrode_config_name': electrode_config_name})
        probe.ElectrodeConfig.Electrode.insert({**electrode_config_key, **electrode}
                                               for electrode in electrodes)

    return electrode_config_key


def get_recording_channels_details(ephys_recording_key):
    channels_details = {}

    acq_software, sample_rate = (EphysRecording & ephys_recording_key).fetch1('acq_software',
                                                              'sampling_rate')

    probe_type = (ProbeInsertion * probe.Probe & ephys_recording_key).fetch1('probe_type')
    channels_details['probe_type'] = {'neuropixels 1.0 - 3A': '3A',
                                      'neuropixels 1.0 - 3B': 'NP1',
                                      'neuropixels UHD': 'NP1100',
                                      'neuropixels 2.0 - SS': 'NP21',
                                      'neuropixels 2.0 - MS': 'NP24'}[probe_type]

    electrode_config_key = (probe.ElectrodeConfig * EphysRecording & ephys_recording_key).fetch1('KEY')
    channels_details['channel_ind'], channels_details['x_coords'], channels_details[
        'y_coords'], channels_details['shank_ind'] = (
            probe.ElectrodeConfig.Electrode * probe.ProbeType.Electrode
            & electrode_config_key).fetch('electrode', 'x_coord', 'y_coord', 'shank')
    channels_details['sample_rate'] = sample_rate
    channels_details['num_channels'] = len(channels_details['channel_ind'])

    if acq_software == 'SpikeGLX':
        spikeglx_meta_filepath = get_spikeglx_meta_filepath(ephys_recording_key)
        spikeglx_recording = spikeglx.SpikeGLX(spikeglx_meta_filepath.parent)
        channels_details['uVPerBit'] = spikeglx_recording.get_channel_bit_volts('ap')[0]
        channels_details['connected'] = np.array(
            [v for *_, v in spikeglx_recording.apmeta.shankmap['data']])
    elif acq_software == 'Open Ephys':
        oe_probe = get_openephys_probe_data(ephys_recording_key)
        channels_details['uVPerBit'] = oe_probe.ap_meta['channels_gains'][0]
        channels_details['connected'] = np.array([
            int(v == 1) for c, v in oe_probe.channels_connected.items()
            if c in channels_details['channel_ind']])

    return channels_details
