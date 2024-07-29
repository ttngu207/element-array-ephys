from __future__ import annotations

import os
import datajoint as dj
from datetime import datetime, timezone
from pathlib import Path
import json
import shutil

from element_interface.utils import dict_to_uuid, find_full_path

logger = dj.logger

schema = dj.schema()

ephys = None
ephys_sorter = None


def activate(
        schema_name,
        *,
        ephys_module,
        ephys_sorter_module,
        create_schema=True,
        create_tables=True,
):
    """
    activate(schema_name, *, ephys_module, ephys_sorter_module, create_schema=True, create_tables=True)
        :param schema_name: schema name on the database server to activate the `ephys_curation` schema
        :param ephys_module: the activated ephys module for which this `ephys_curation` schema will be downstream from
        :param ephys_sorter_module: the activated ephys_sorter module for which this `ephys_curation` schema will be downstream from
        :param create_schema: when True (default), create schema in the database if it does not yet exist.
        :param create_tables: when True (default), create tables in the database if they do not yet exist.
    """
    global ephys, ephys_sorter
    ephys = ephys_module
    ephys_sorter = ephys_sorter_module
    schema.activate(
        schema_name,
        create_schema=create_schema,
        create_tables=create_tables,
        add_objects={**ephys.__dict__, **ephys_sorter.__dict__},
    )


@schema
class CurationMethod(dj.Lookup):
    definition = """
    # Curation method
    curation_method: varchar(16)  # method/package used to perform manual curation (e.g. Phy, FigURL, etc.)
    """
    contents = [
        ("Phy",),
    ]


@schema
class ManualCuration(dj.Manual):
    definition = """
    # Manual curation from an ephys.Clustering
    -> ephys.Clustering
    curation_id=0: int
    ---
    curation_datetime: datetime    # UTC time when the curation was performed
    parent_curation_id=-1: int     # if -1, this curation is based on the raw spike sorting results
    -> CurationMethod              # which method/package used for manual curation (inform how to ingest the results)
    description="": varchar(1000)  # user-defined description/note of the curation
    """

    class File(dj.Part):
        definition = """
        -> master
        file_id: int
        ---
        file_name: varchar(1000)
        file: filepath@ephys-store
        """

    @classmethod
    def prepare_manual_curation(cls, key, *, parent_curation_id=-1, curation_method="Phy", download_raw=False):
        """
        Create a new directory to for a new round of manual curation
        Download the spike sorting results for new manual curation from the specified "key" and "parent_curation_id".
        Store the initial meta information in json file.
        Args:
            key: PK of the ephys.Clustering table
            parent_curation_id:
            curation_method: method/package used for manual curation
            download_raw: if True, also download the raw ephys (.dat) file

        Returns:
            directory where the spike sorting results are downloaded
        """
        if curation_method != "Phy":
            raise ValueError(f"Unsupported curation method: {curation_method}")

        init_datetime = datetime.now(timezone.utc)

        # Download the spike sorting results
        assert ephys.Clustering & key, f"Invalid ephys.Clustering key: {key}"

        if parent_curation_id == -1:
            assert ephys_sorter.SIExport & key, "SIExport not found for the specified key"
            files_query = ephys_sorter.SIExport.File & key & "file_name LIKE 'phy%' AND NOT LIKE '%recording.dat'"
        else:
            assert cls & {**key, "curation_id": parent_curation_id}, "ManualCuration not found for the specified key"
            files_query = cls.File & {**key, "curation_id": parent_curation_id}

        # create new directory for new curation
        output_dir = (
                ephys.ClusteringTask & key
        ).fetch1("clustering_output_dir")
        output_dir = find_full_path(ephys.get_ephys_root_data_dir(), output_dir)

        dirname = f"{curation_method}_curation" + ("" if parent_curation_id == -1 else f"_parent_{parent_curation_id}")
        curation_output_dir = output_dir / dirname
        curation_output_dir.mkdir(parents=True, exist_ok=True)

        # check for existing curation directory
        if (curation_output_dir / ".curation_meta.json").exists():
            with open(curation_output_dir / ".curation_meta.json", "r") as f:
                meta = json.load(f)
            status = meta["status"]
            logger.info(f"Existing curation directory found: {curation_output_dir} - status: {status}")
        else:
            # download spike sorting files and copy to the new directory
            logger.info(f"New manual curation: {curation_output_dir} - Downloading {len(files_query)} files...")

            files = list(files_query.fetch("file"))
            if download_raw:
                f = (ephys_sorter.SIExport.File & key & "file_name = 'phy/recording.dat'").fetch1("file")
                files.append(f)
            for f in files:
                new_f = curation_output_dir / f.relative_to(output_dir)
                new_f.write_bytes(f.read_bytes())

            # write a json file with the initial meta information
            with open(curation_output_dir / ".curation_meta.json", "w") as f:
                json.dump(
                    {
                        "parent_table": cls.full_table_name if parent_curation_id != -1 else ephys_sorter.SIExport.full_table_name,
                        "key": key,
                        "parent_curation_id": parent_curation_id,
                        "curation_method": curation_method,
                        "init_datetime": init_datetime.strftime('%Y%m%d_%H%M%S'),
                        "status": "initialized"
                    },
                    f,
                )

        return curation_output_dir

    @classmethod
    def insert_curation(cls, key, *, curation_output_dir, parent_curation_id=-1, curation_method="Phy", description=""):
        if curation_method != "Phy":
            raise ValueError(f"Unsupported curation method: {curation_method}")

        from element_array_ephys.readers import kilosort
        kilosort_dataset = kilosort.Kilosort(curation_output_dir)
        assert kilosort_dataset.data, f"Invalid Phy output directory: {curation_output_dir}"

        curate_datetime = datetime.now(timezone.utc)
        curation_id = (cls & key).fetch("curation_id").max() + 1
        logger.info(f"New curation id: {curation_id}")

        key["curation_id"] = curation_id
        with cls.connection.transaction:
            cls.insert1(
                {
                    **key,
                    "curation_datetime": curate_datetime,
                    "parent_curation_id": parent_curation_id,
                    "curation_method": curation_method,
                    "description": description,
                }
            )

            # rename curation_output_dir folder into curation_id
            new_curation_output_dir = curation_output_dir.parent / f"{curation_method}_curation_{curation_id}"
            curation_output_dir.rename(new_curation_output_dir)

            cls.File.insert(
                [
                    {**key, "file_id": i, "file_name": f.relative_to(new_curation_output_dir).name, "file": f}
                    for i, f in enumerate(new_curation_output_dir.rglob("*"))
                    if f.is_file() and f.name != "recording.dat"
                ]
            )

        logger.info(f"New manual curation inserted: {key}")

        return key


def commit_new_curation(curation_output_dir, description="", do_insert=True):
    if not curation_output_dir.is_dir():
        raise FileNotFoundError(f"Invalid curation directory: {curation_output_dir}")

    with open(curation_output_dir / ".curation_meta.json", "r") as f:
        meta = json.load(f)

    status = meta["status"]
    if status == "inserted":
        logger.info(f"Curation already inserted: {meta}")
        return meta

    meta.update({"status": "committed", "description": description})
    with open(curation_output_dir / ".curation_meta.json", "w") as f:
        json.dump(meta, f)

    if do_insert:
        assert status == "committed", "Must be committed before insert"
        key = ManualCuration.insert_curation(key=meta["key"],
                                             curation_output_dir=curation_output_dir,
                                             parent_curation_id=meta["parent_curation_id"],
                                             curation_method=meta["curation_method"],
                                             description=meta["description"])

        meta.update({**key, "status": "inserted"})
        with open(curation_output_dir / ".curation_meta.json", "w") as f:
            json.dump(meta, f)

    return meta


def launch_phy(key, parent_curation_id=-1, download_raw=False, do_insert=True):
    """
    Select a spike sorting key for manual curation
    1. download the spike sorting results
    2. launch phy
    3. commit new curation locally
    4. insert new curation into the database
    Args:
        key: ephys.Clustering key
        parent_curation_id: if -1, this curation is based on the raw spike sorting results
        download_raw: if True, also download the raw ephys (.dat) file
        do_insert: if True, insert the new curation into the database (upload result files)
    """
    from phy.apps.template import template_gui

    curation_output_dir = ManualCuration.prepare_manual_curation(key,
                                                                 parent_curation_id=parent_curation_id,
                                                                 download_raw=download_raw)

    template_gui(curation_output_dir / "params.py")

    if dj.utils.user_choice('Commit new manual curation?') != 'yes':
        print('Canceled')
        return

    description = input('Curation description: ')

    commit_new_curation(curation_output_dir, description=description, do_insert=do_insert)
