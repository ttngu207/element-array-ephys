import datajoint as dj
import os
import logging
from . import clustering

try:
    from phy.apps.template import TemplateController, template_gui
    from phy.gui.qt import create_app, run_app
    from phylib import add_default_handler
except ModuleNotFoundError:
    raise ModuleNotFoundError('Phy is not yet installed. Follow installation instruction at https://github.com/cortex-lab/phy')


def launch_phy(key):
    """
    Launch phy for a particular Clustering or Curation
    :param: key - either a Clustering key or a Curation key
    """
    table = clustering.CurationFile if 'curation_id' in key else clustering.ClusteringFile
    assert len(table & key) == 1, f'Invalid key for {table.__name__}: {key}'

    print(f'Preparing to launch phy for {table.__name__}: {key}')

    # download files
    raw_file, clustering_dir = clustering.fetch_files(key)
    # navigate to the directory containing the files
    print(f'\tRaw data file: {raw_file}')
    print(f'\tClustering directory: {clustering_dir}')

    current_dir = os.getcwd()
    os.chdir(clustering_dir)

    # -------- Launch phy ------------
    print('\tLaunching phy...')
    add_default_handler('DEBUG', logging.getLogger("phy"))
    add_default_handler('DEBUG', logging.getLogger("phylib"))
    create_app()
    controller = TemplateController(dat_path=raw_file, dir_path=clustering_dir)
    gui = controller.create_gui()
    gui.show()
    run_app()
    gui.close()
    controller.model.close()

    clustering.create_new_curation(key, clustering_dir=clustering_dir, prompt=True)

    os.chdir(current_dir)
