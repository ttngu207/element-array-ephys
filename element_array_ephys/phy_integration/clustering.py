import datajoint as dj
import importlib
import inspect
import pathlib
import shutil

from element_interface.utils import find_full_path, find_root_directory

schema = dj.schema()

ephys = None


def activate(ephys_schema_name, *, create_schema=True,
             create_tables=True, linking_module=None):

    if isinstance(linking_module, str):
        linking_module = importlib.import_module(linking_module)
    assert inspect.ismodule(linking_module),\
        "The argument 'dependency' must be a module's name or a module"

    global ephys
    ephys = linking_module.ephys

    schema.activate(ephys_schema_name, create_schema=create_schema,
                    create_tables=create_tables, add_objects=linking_module.__dict__)


@schema
class EphysFile(dj.Imported):
    definition = """
    -> ephys.EphysRecording
    file_name: varchar(64)  # filepath.stem
    ---
    file_path: filepath@ephys_raw
    """

    @property
    def key_source(self):
        return ephys.EphysRecording()

    def make(self, key):
        acq_software = (ephys.EphysRecording & key).fetch1('acq_software')
        ephys_file = (ephys.EphysRecording.EphysFile & key).fetch('file_path', limit=1)[0]
        ephys_file = find_full_path(ephys.get_ephys_root_data_dir(), ephys_file)
        if ephys_file.is_dir():
            ephys_dir = ephys_file
        elif ephys_file.is_file():
            ephys_dir = ephys_file.parent
        else:
            raise ValueError(f'{ephys_file} is either file or directory')

        if acq_software == 'SpikeGLX':
            raw_file = ephys_dir / f'{ephys_file.stem}.bin'
        elif acq_software == 'Open Ephys':
            raw_file = ephys_dir / 'continuous.dat'
        else:
            raise NotImplementedError(f'Unknown acquisition software: {acq_software}')

        self.insert1({**key, 'file_name': f'{raw_file.parent.name}/{raw_file.name}', 'file_path': raw_file})


@schema
class ClusteringFile(dj.Imported):
    definition = """
    -> ephys.Clustering
    file_name: varchar(64)  # filepath.stem
    ---
    file_path: filepath@ephys_processed
    """

    @property
    def key_source(self):
        return ephys.Clustering()

    def make(self, key):
        output_dir = (ephys.ClusteringTask & key).fetch1('clustering_output_dir')
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        files = [{**key, 'file_name': f.name, 'file_path': f}
                 for f in output_dir.glob('*') if f.is_file()]
        self.insert(files)


class CurationFile(dj.Imported):
    definition = """
    -> ephys.Curation
    file_name: varchar(64)  # filepath.stem
    ---
    file_path: filepath@ephys_processed
    """

    @property
    def key_source(self):
        return ephys.Curation()

    def make(self, key):
        output_dir = (ephys.Curation & key).fetch1('curation_output_dir')
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        files = [{**key, 'file_name': f.name, 'file_path': f}
                 for f in output_dir.glob('*') if f.is_file()]
        self.insert(files)


def fetch_files(key):
    """
    Fetch all resulting files from a Clustering or Curation
    :param: key - either a Clustering key or a Curation key
    """
    raw_file = (EphysFile & key).fetch('file_path', limit=1)[0]

    table = CurationFile if 'curation_id' in key else ClusteringFile
    clustering_files = (table & key).fetch('file_path')
    print(f'\t{len(clustering_files)} files fetched')
    return pathlib.Path(raw_file), pathlib.Path(clustering_files[0]).parent


def create_new_curation(key, clustering_dir, curation_id=None, curation_note=''):
    clustering_dir = pathlib.Path(clustering_dir)
    assert clustering_dir.exists()

    clustering_method = (ephys.Clustering * ephys.ClusteringParamSet & key).fetch1('clustering_method')

    assert 'kilosort' in clustering_method, f'Unable to handle clustering method: {clustering_method}'

    from ..readers import kilosort
    creation_time, is_curated, is_qc = kilosort.extract_clustering_info(clustering_dir)

    if curation_id is None:
        curation_id = dj.U().aggr(ephys.Curation & key, n='ifnull(max(curation_id)+1,1)').fetch1('n')

    root_dir = find_root_directory(ephys.get_ephys_root_data_dir(), clustering_dir)

    curation_dir = clustering_dir / f'curation_{curation_id}'
    # curation_dir.mkdir(parents=True, exist_ok=True)

    curation_key = {**key, 'curation_id': curation_id}

    with ephys.Curation.connection.transaction:
        ephys.Curation.insert1({**curation_key,
                                'curation_time': creation_time,
                                'curation_output_dir': curation_dir.relative_to(root_dir).as_posix(),
                                'quality_control': is_qc,
                                'manual_curation': is_curated,
                                'curation_note': f'Curation based on: {key}\n' + curation_note})
        # ---- upload files ----
        shutil.copytree(str(clustering_dir), str(curation_dir))
        CurationFile().make(curation_key)
