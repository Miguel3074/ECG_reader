import os
import uuid
from flask import Flask, render_template, request, redirect, url_for, flash
import ecg_analyzer  # Importa seu módulo de análise
import json
import numpy as np
import cv2  # Importe a biblioteca OpenCV

app = Flask(__name__)
app.secret_key = os.environ.get(
    "SESSION_SECRET",
    "dev_key_for_ecg_analyzer")  # Chave secreta para flash messages
UPLOAD_FOLDER = 'temp_uploads'  # Defina sua pasta de uploads
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
# Permitir PNG e JPG/JPEG
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
# Criar a pasta de uploads se não existir
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config[
    'MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # Limite de 16MB para upload


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    if 'ecg_image' not in request.files:
        flash('Nenhum arquivo enviado', 'danger')
        return redirect(request.url)  # Redireciona para a mesma página (index)

    file = request.files['ecg_image']

    if file.filename == '':
        flash('Nenhum arquivo selecionado', 'danger')
        return redirect(url_for('index'))  # Redireciona para o início

    if not allowed_file(file.filename):
        # Informa quais extensões são permitidas
        allowed_ext_str = ', '.join(ALLOWED_EXTENSIONS)
        flash(f'Tipo de arquivo não permitido. Use: {allowed_ext_str}',
              'danger')
        return redirect(url_for('index'))

    filepath = None  # Inicializa filepath com None
    try:
        file_ext = ''  # Inicializa file_ext com um valor padrão
        if file.filename:
            file_ext = file.filename.rsplit('.', 1)[1].lower()
        unique_filename = str(uuid.uuid4()) + '.' + file_ext
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(filepath)

        default_sample_rate_hz = 95.0  # Estimativa de pixels/segundo

        # Definindo a ROI para ser a imagem inteira (y_start, y_end, x_start, x_end)
        # Precisamos carregar a imagem para obter suas dimensões
        img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if img is None:
            flash('Erro ao carregar a imagem para obter dimensões.', 'danger')
            return redirect(url_for('index'))
        height, width = img.shape
        roi_coords = (0, height, 0, width)

        # Chamar a função de análise COM as coordenadas da ROI
        results = ecg_analyzer.analyze_ecg(filepath, roi_coords,
                                           default_sample_rate_hz)

        # Salvar resultados em JSON temporário para passar para a próxima rota
        result_filename = str(uuid.uuid4()) + '.json'
        result_filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                       result_filename)

        with open(result_filepath, 'w') as result_file:
            # Converter numpy arrays (se houver) para listas para serialização JSON
            if 'rr_intervals_ms' in results and isinstance(
                    results['rr_intervals_ms'], np.ndarray):
                results['rr_intervals_ms'] = results['rr_intervals_ms'].tolist(
                )
            json.dump(results, result_file)

        # Remover o arquivo de imagem original após o processamento
        try:
            os.remove(filepath)
        except OSError as e:
            app.logger.warning(
                f"Não foi possível remover o arquivo de imagem temporário {filepath}: {e}"
            )

        # Redirecionar para a página de resultados, passando o nome do arquivo JSON
        return redirect(url_for('results', result_filename=result_filename))

    except Exception as e:
        # Logar o erro completo no console do Flask para depuração
        app.logger.error(f'Erro durante o processamento do ECG: {str(e)}',
                         exc_info=True)
        # Mostrar uma mensagem genérica para o usuário
        flash(
            f'Ocorreu um erro inesperado ao processar o ECG. Verifique o log do servidor.',
            'danger')

        # Tentar remover o arquivo de imagem se ele existir e um erro ocorreu
        if filepath and os.path.exists(
                filepath):  # Verifica se filepath não é None
            try:
                os.remove(filepath)
            except OSError as e_rem:
                app.logger.warning(
                    f"Não foi possível remover o arquivo de imagem {filepath} após erro: {e_rem}"
                )

        return redirect(url_for('index'))


@app.route('/results')
def results():
    result_filename = request.args.get('result_filename')
    if not result_filename or '..' in result_filename or result_filename.startswith(
            '/'):
        flash('Nome de arquivo de resultado inválido.', 'danger')
        return redirect(url_for('index'))

    result_filepath = os.path.join(app.config['UPLOAD_FOLDER'],
                                   result_filename)

    if not os.path.exists(result_filepath):
        flash(
            'Arquivo de resultados não encontrado ou expirado. Por favor, tente novamente.',
            'danger')
        return redirect(url_for('index'))

    try:
        with open(result_filepath, 'r') as result_file:
            results_data = json.load(result_file)

        # Remover o arquivo JSON após carregá-lo
        try:
            os.remove(result_filepath)
        except OSError as e:
            app.logger.warning(
                f"Não foi possível remover o arquivo de resultado {result_filepath}: {e}"
            )

        # Passar os dados carregados para o template
        return render_template('results.html', results=results_data)

    except json.JSONDecodeError:
        flash('Erro ao ler o arquivo de resultados. Formato inválido.',
              'danger')
        if os.path.exists(result_filepath):
            os.remove(result_filepath)  # Tenta remover mesmo se corrompido
        return redirect(url_for('index'))
    except Exception as e:
        app.logger.error(f'Erro ao carregar resultados: {str(e)}',
                         exc_info=True)
        flash('Erro ao carregar a página de resultados.', 'danger')
        if os.path.exists(result_filepath):
            os.remove(result_filepath)  # Tenta remover mesmo se corrompido
        return redirect(url_for('index'))


if __name__ == '__main__':
    # host='0.0.0.0' permite acesso de outros dispositivos na rede
    # debug=True é útil para desenvolvimento, mas DESATIVE em produção
    app.run(host='0.0.0.0', port=5000, debug=True)
