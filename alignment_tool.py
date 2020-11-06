import numpy as np
import scipy
#import tempfile
#import subprocess
#import os
#import pretty_midi
import librosa
import djitw

AUDIO_FS = 32000
AUDIO_HOP = 384
NOTE_START = 24
N_NOTES = 48


def audio_cqt(audio_data, 
              fs=AUDIO_FS, 
              n_stt=NOTE_START, 
              n_nts=N_NOTES,
              a_hop=AUDIO_HOP):
    '''
    Compute some audio data's constant-Q spectrogram, normalize, and log-scale
    it

    Parameters
    ----------
    audio_data : np.ndarray
        Some audio signal.
    fs : int
        Sampling rate the audio data is sampled at, should be ``AUDIO_FS``.

    Returns
    -------
    midi_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrugram of synthesized MIDI
        data.
    '''
    # Compute CQT of the synthesized audio data
    audio_gram = librosa.cqt(audio_data, 
                             sr=fs, 
                             hop_length=a_hop,
                             fmin=librosa.midi_to_hz(n_stt), 
                             n_bins=n_nts)
    
    # L2-normalize and log-magnitute it
    out_data = post_process_cqt(audio_gram)
    
    return out_data


def post_process_cqt(gram):
    '''
    Normalize and log-scale a Constant-Q spectrogram

    Parameters
    ----------
    gram : np.ndarray
        Constant-Q spectrogram, constructed from ``librosa.cqt``.

    Returns
    -------
    log_normalized_gram : np.ndarray
        Log-magnitude, L2-normalized constant-Q spectrogram.
    '''
    # Compute log amplitude
    #gram = librosa.amplitude_to_db(np.abs(gram), top_db=np.abs(gram).max())
    gram = (librosa.amplitude_to_db(np.abs(gram), amin=1e-06, top_db=80.0) + 80.001) * (100.0/80.0)
    # Transpose so that rows are samples
    gram = gram.T
    # and L2 normalize
    gram = librosa.util.normalize(gram, norm=2., axis=1)
    # and convert to float32
    return gram.astype(np.float32)


def get_cqt_by_adata(audio_data_in_tmp, 
                     samp_rate_in_tmp=AUDIO_FS, 
                     hop_in_tmp=AUDIO_HOP,
                     note_stt_in_tmp=NOTE_START,
                     n_notes_in_tmp=N_NOTES):
       
    cqt_data_out_tmp_final = audio_cqt(audio_data_in_tmp, 
                                       fs=samp_rate_in_tmp, 
                                       n_stt=note_stt_in_tmp, 
                                       n_nts=n_notes_in_tmp,
                                       a_hop=hop_in_tmp)
    return cqt_data_out_tmp_final


def check_path_validity(midx, audx, hop_size=AUDIO_HOP, samp_rate=AUDIO_FS):
    
    rmv_f = int(3.0/(hop_size/samp_rate))
    
    slope_v,    \
    intrcpt,    \
    r_value,    \
    p_value,    \
    std_err = scipy.stats.linregress(midx[rmv_f:-rmv_f], 
                                     audx[rmv_f:-rmv_f])
    
    if not (0.97<=slope_v<=1.03): 
        return False
    elif not (-25<=intrcpt<=25):
        return False
    else: 
        return True
    
    
def alignment_func(audio_gram1, 
                   audio_gram2, 
                   hv_pen_rto,
                   gully_rto,
                   ):

    distance_matrix = 1.0 - np.dot(audio_gram2, audio_gram1.T)

    add_pen_v = np.median(distance_matrix) * hv_pen_rto

    aligned_audio2_indices, aligned_audio1_indices, dtw_score = djitw.dtw(distance_matrix, 
                                                                          gully=gully_rto, 
                                                                          additive_penalty=add_pen_v,
                                                                          inplace=False)
    
    
    # Normalize score by path length
    dtw_score /= float(len(aligned_audio2_indices))
    
    
    # Normalize score by score by mean sim matrix value within path chunk
    dtw_score /= distance_matrix[aligned_audio2_indices.min():aligned_audio2_indices.max(),
                                 aligned_audio1_indices.min():aligned_audio1_indices.max()].mean()
            
    #return score
    return aligned_audio1_indices, aligned_audio2_indices, dtw_score
