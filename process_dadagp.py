#!/usr/bin/env python3
import os
import subprocess
import tempfile
from io import BytesIO
from flask import Flask, request, send_file, abort

app = Flask(__name__)

# Configuration – adjust these paths as necessary:
TUXGUITAR_CMD = 'tuxguitar'  # Ensure TuxGuitar is in your PATH, or use the full path.
FLUIDSYNTH_CMD = 'fluidsynth'  # Ensure FluidSynth is installed.
SOUNDFONT_PATH = '/path/to/your/soundfont.sf2'  # Update this to the location of your .sf2 soundfont.

@app.route('/')
def index():
    return '''
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8">
      <title>GP3 to Waveform Conversion</title>
    </head>
    <body>
      <h1>Upload a .gp3 File for Conversion</h1>
      <form method="post" action="/convert" enctype="multipart/form-data">
        <input type="file" name="file" accept=".gp3">
        <input type="submit" value="Upload">
      </form>
    </body>
    </html>
    '''

@app.route('/convert', methods=['POST'])
def convert():
    if 'file' not in request.files:
        abort(400, 'No file part in the request.')
    file = request.files['file']
    if file.filename == '':
        abort(400, 'No file selected.')
    if not file.filename.lower().endswith('.gp3'):
        abort(400, 'The uploaded file is not a .gp3 file.')

    # Save the uploaded .gp3 file to a temporary location.
    with tempfile.NamedTemporaryFile(suffix='.gp3', delete=False) as temp_gp3:
        file.save(temp_gp3.name)
        gp3_path = temp_gp3.name

    # Create temporary files for the intermediate MIDI and final WAV files.
    midi_fd, midi_path = tempfile.mkstemp(suffix='.mid')
    os.close(midi_fd)  # Close the file descriptor; we only need the path.
    wav_fd, wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(wav_fd)

    try:
        # Convert .gp3 to MIDI using TuxGuitar.
        # The command below assumes TuxGuitar supports command-line export.
        # You may need to adjust the arguments depending on your TuxGuitar installation.
        tg_command = [TUXGUITAR_CMD, '--export-midi', gp3_path, midi_path]
        subprocess.run(tg_command, check=True)
        
        # Convert the MIDI file to a WAV file using FluidSynth.
        fs_command = [
            FLUIDSYNTH_CMD, '-ni', SOUNDFONT_PATH, midi_path,
            '-F', wav_path, '-r', '44100'
        ]
        subprocess.run(fs_command, check=True)
        
        # Read the WAV file into memory.
        with open(wav_path, 'rb') as f:
            wav_data = f.read()

        # Serve the WAV file as a downloadable attachment.
        return send_file(
            BytesIO(wav_data),
            as_attachment=True,
            download_name='output.wav',
            mimetype='audio/wav'
        )
    
    except subprocess.CalledProcessError as e:
        return f"Conversion failed: {e}", 500

    finally:
        # Clean up temporary files.
        for path in [gp3_path, midi_path, wav_path]:
            try:
                os.remove(path)
            except OSError:
                pass

if __name__ == '__main__':
    # Run the Flask app – for production use, consider a production server.
    app.run(debug=True)