from .. import ephys_acute, ephys_chronic, ephys_no_curation
from . import clustering

for ephys in (ephys_acute, ephys_chronic, ephys_no_curation):
    if ephys.schema.is_activated():
        if hasattr(ephys, 'Curation'):
            clustering.schema(clustering.CurationFile)
        clustering.activate(ephys.schema.database, linking_module=__name__)
        break
else:
    raise AssertionError('ephys not yet activated')
