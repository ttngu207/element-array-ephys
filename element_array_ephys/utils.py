class Makes():
    
    @staticmethod
    def make_ephys_recording(table, key):
        sess_dir = find_full_path(get_ephys_root_data_dir(),
                                  get_session_directory(key))
        inserted_probe_serial_number = (ProbeInsertion * probe.Probe & key).fetch1('probe')

        # search session dir and determine acquisition software
        for ephys_pattern, ephys_acq_type in zip(['*.ap.meta', '*.oebin'],
                                                 ['SpikeGLX', 'Open Ephys']):
            ephys_meta_filepaths = list(sess_dir.rglob(ephys_pattern))
            if ephys_meta_filepaths:
                acq_software = ephys_acq_type
                break
        else:
            raise FileNotFoundError(
                f'Ephys recording data not found!'
                f' Neither SpikeGLX nor Open Ephys recording files found')

        supported_probe_types = probe.ProbeType.fetch('probe_type')

        if acq_software == 'SpikeGLX':
            for meta_filepath in ephys_meta_filepaths:
                spikeglx_meta = spikeglx.SpikeGLXMeta(meta_filepath)
                if str(spikeglx_meta.probe_SN) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No SpikeGLX data found for probe insertion: {}'.format(key))

            if spikeglx_meta.probe_model in supported_probe_types:
                probe_type = spikeglx_meta.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {
                    (shank, shank_col, shank_row): key
                    for key, shank, shank_col, shank_row in zip(*electrode_query.fetch(
                        'KEY', 'shank', 'shank_col', 'shank_row'))}

                electrode_group_members = [
                    probe_electrodes[(shank, shank_col, shank_row)]
                    for shank, shank_col, shank_row, _ in spikeglx_meta.shankmap['data']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels probe model'
                    ' {} not yet implemented'.format(spikeglx_meta.probe_model))

            table.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': spikeglx_meta.meta['imSampRate'],
                'recording_datetime': spikeglx_meta.recording_time,
                'recording_duration': (spikeglx_meta.recording_duration
                                       or spikeglx.retrieve_recording_duration(meta_filepath))})

            root_dir = find_root_directory(get_ephys_root_data_dir(), meta_filepath)
            table.EphysFile.insert1({
                **key,
                'file_path': meta_filepath.relative_to(root_dir).as_posix()})
        elif acq_software == 'Open Ephys':
            dataset = openephys.OpenEphys(sess_dir)
            for serial_number, probe_data in dataset.probes.items():
                if str(serial_number) == inserted_probe_serial_number:
                    break
            else:
                raise FileNotFoundError(
                    'No Open Ephys data found for probe insertion: {}'.format(key))

            if probe_data.probe_model in supported_probe_types:
                probe_type = probe_data.probe_model
                electrode_query = probe.ProbeType.Electrode & {'probe_type': probe_type}

                probe_electrodes = {key['electrode']: key
                                    for key in electrode_query.fetch('KEY')}

                electrode_group_members = [
                    probe_electrodes[channel_idx]
                    for channel_idx in probe_data.ap_meta['channels_indices']]
            else:
                raise NotImplementedError(
                    'Processing for neuropixels'
                    ' probe model {} not yet implemented'.format(probe_data.probe_model))

            table.insert1({
                **key,
                **generate_electrode_config(probe_type, electrode_group_members),
                'acq_software': acq_software,
                'sampling_rate': probe_data.ap_meta['sample_rate'],
                'recording_datetime': probe_data.recording_info['recording_datetimes'][0],
                'recording_duration': np.sum(probe_data.recording_info['recording_durations'])})

            root_dir = find_root_directory(
                get_ephys_root_data_dir(),
                probe_data.recording_info['recording_files'][0])
            table.EphysFile.insert([{**key,
                                    'file_path': fp.relative_to(root_dir).as_posix()}
                                   for fp in probe_data.recording_info['recording_files']])
        else:
            raise NotImplementedError(f'Processing ephys files from'
                                      f' acquisition software of type {acq_software} is'
                                      f' not yet implemented')
