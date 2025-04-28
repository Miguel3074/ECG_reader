import os
import cv2
import numpy as np
import csv
import svgwrite
from flask import Flask, jsonify, render_template, request, redirect, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
import traceback
import ecg_analyzer


app = Flask(__name__)
STATIC_FOLDER = 'static'
UPLOAD_FOLDER = os.path.join(STATIC_FOLDER, 'uploads')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}
app.secret_key = os.urandom(24)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def remove_grid_and_extract_points(img_bgr):
    print("Removendo grade (Inpainting) e extraindo todos os pontos brancos...")
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    lower_red1 = np.array([0, 70, 50])
    upper_red1 = np.array([10, 255, 255])
    mask1 = cv2.inRange(img_hsv, lower_red1, upper_red1)
    lower_red2 = np.array([170, 70, 50])
    upper_red2 = np.array([180, 255, 255])
    mask2 = cv2.inRange(img_hsv, lower_red2, upper_red2)
    grid_mask = mask1 | mask2
    img_no_grid = cv2.inpaint(img_bgr, grid_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
    gray = cv2.cvtColor(img_no_grid, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY_INV)
    cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], 'debug_global_thresh.png'), thresh)
    print("Extraindo coordenadas de todos os pixels brancos...")
    all_points = []
    height, width = thresh.shape
    for y in range(height):
        for x in range(width):
            if thresh[y, x] == 255:
                all_points.append((x, y))
    print(f"Total de pontos brancos extraídos: {len(all_points)}")
    if not all_points:
        print("Nenhum ponto branco encontrado na imagem binarizada.")
        return None, None
    dimensions = (width, height)
    return all_points, dimensions

def salvar_pontos_csv(points, csv_filepath):
    """Salva a lista de pontos (x, y) em um arquivo CSV, ordenados por Y e depois por X."""
    print(f"Salvando pontos no CSV (ordenado por Y, então X): {csv_filepath}")
    try:
        points_ordenados = sorted(points, key=lambda p: (p[1], p[0]))
        with open(csv_filepath, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['x', 'y'])
            for point in points_ordenados:
                writer.writerow(point)
        print("Pontos salvos e ordenados com sucesso.")
    except Exception as e:
        print(f"Erro ao salvar CSV: {e}")

def gerar_svg_a_partir_dos_pontos(points, dimensions, svg_filepath):
    """Gera um SVG mostrando todos os pontos e conectando os próximos (mesma cor, tamanhos maiores)."""
    if not points:
        print("Nenhum ponto fornecido para gerar SVG.")
        return False
    print(f"Gerando SVG com todos os pontos e conexões por proximidade (raio=10, cor única, tamanhos maiores): {svg_filepath}")
    print(f"Número de pontos recebidos: {len(points)}")
    if not all(isinstance(p, tuple) and len(p) == 2 and all(isinstance(coord, (int, float, np.number)) for coord in p) for p in points[:10]):
        print("Formato de pontos inválido detectado na verificação inicial. Abortando geração do SVG.")
        return False
    try:
        width, height = dimensions
        dwg = svgwrite.Drawing(svg_filepath, profile='tiny', size=(f"{width}px", f"{height}px"))
        dwg.viewbox(0, 0, width, height)
        cor = 'blue'
        stroke_width_val = 1.0
        point_radius = 0.5
        print("Ordenando pontos por coordenada X...")
        points.sort(key=lambda p: p[0])
        print("Adicionando todos os pontos como pequenos círculos...")
        for x, y in points:
            dwg.add(dwg.circle(center=(x, y), r=point_radius, fill=cor))
        print("Gerando polilinhas conectando pontos próximos...")
        polylines = []
        current_polyline = []
        for i, p in enumerate(points):
            x, y = int(p[0]), int(p[1])
            if not current_polyline:
                current_polyline.append((x, y))
            else:
                last_x, last_y = current_polyline[-1]
                distance = np.sqrt((x - last_x)**2 + (y - last_y)**2)
                if distance <= 10:
                    current_polyline.append((x, y))
                else:
                    if len(current_polyline) > 1:
                        polylines.append(current_polyline)
                    current_polyline = [(x, y)]
        if len(current_polyline) > 1:
            polylines.append(current_polyline)
        print(f"Geradas {len(polylines)} polilinhas.")
        for polyline_points in polylines:
            dwg.add(dwg.polyline(points=polyline_points, fill='none', stroke=cor, stroke_width=stroke_width_val))
        dwg.save()
        print("SVG gerado com sucesso (mesma cor, tamanhos maiores).")
        return True
    except Exception as e:
        print(f"Erro ao gerar SVG (mesma cor, tamanhos maiores): {e}")
        print("--- Traceback ---")
        print(traceback.format_exc())
        print("--- Fim Traceback ---")
        return False

# --- Rotas Flask ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'ecg_image' not in request.files:
        flash('Nenhuma parte do arquivo')
        return redirect(request.url)
    file = request.files['ecg_image']
    if file.filename == '':
        flash('Nenhum arquivo selecionado')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        if not os.path.exists(app.config['UPLOAD_FOLDER']):
            os.makedirs(app.config['UPLOAD_FOLDER'])
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        print(f"Arquivo salvo em: {filepath}")
        img = cv2.imread(filepath)
        if img is None:
            flash('Erro ao carregar a imagem.')
            return redirect(request.url)
        ecg_points, dimensions = remove_grid_and_extract_points(img)
        if ecg_points is None or dimensions is None:
            flash('Não foi possível extrair os pontos do ECG da imagem.')
            return redirect(url_for('index'))
        base_filename = filename.rsplit('.', 1)[0]
        csv_filename = f"{base_filename}_data.csv"
        csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        salvar_pontos_csv(ecg_points, csv_filepath)
        svg_filename = f"{base_filename}_output.svg"
        svg_filepath = os.path.join(app.config['UPLOAD_FOLDER'], svg_filename)
        svg_gerado = gerar_svg_a_partir_dos_pontos(ecg_points, dimensions, svg_filepath)
        if not svg_gerado:
            flash('Erro ao gerar o arquivo SVG.')
            return redirect(url_for('index'))
        original_image_path_relative = os.path.join('uploads', filename).replace('\\', '/')
        svg_path_relative = os.path.join('uploads', svg_filename).replace('\\', '/')
        analysis_results = ecg_analyzer.analyze_ecg(csv_filepath) # Chama a análise no arquivo completo
        return render_template('results.html',
                               results=analysis_results.get("derivations", {}),
                               main_analysis=analysis_results.get("main", {}),
                               svg_image_path=svg_path_relative,
                               csv_download_path=os.path.join('uploads', csv_filename).replace('\\', '/'),
                               original_image_path=original_image_path_relative)
    else:
        flash('Formato de arquivo inválido. Apenas PNG, JPG, JPEG são permitidos.')
        return redirect(request.url)

@app.route('/analyze_segmented', methods=['POST'])
def analyze_segmented():
    try:
        data = request.get_json()
        print(f"Dados recebidos em /analyze_segmented: {data}") # Adicione esta linha
        vertical_lines = sorted(data.get('vertical', []))
        horizontal_lines = sorted(data.get('horizontal', []))
        csv_filename = data.get('csv_filename')

        if not vertical_lines or not horizontal_lines or not csv_filename:
            return jsonify({'error': 'Coordenadas de delimitação ou nome do arquivo CSV inválidos.'})

        # Ler os dados do CSV
        csv_filepath = os.path.join(app.config['UPLOAD_FOLDER'], csv_filename)
        all_points = []
        try:
            with open(csv_filepath, 'r', newline='') as csvfile:
                reader = csv.reader(csvfile)
                next(reader) # Pular o cabeçalho
                for row in reader:
                    try:
                        x = int(row[0])
                        y = int(row[1])
                        all_points.append((x, y))
                    except ValueError:
                        print(f"Erro ao converter linha para inteiro: {row}")
                        continue
        except FileNotFoundError:
            return jsonify({'error': f'Arquivo CSV não encontrado: {csv_filepath}'})
        except Exception as e:
            return jsonify({'error': f'Erro ao ler dados do arquivo CSV: {e}'})

        segmented_derivations = {}
        num_vertical = len(vertical_lines)
        num_horizontal = len(horizontal_lines)

        # Lógica para segmentar os pontos baseados nas linhas
        for i in range(num_horizontal + 1):
            y_min = 0 if i == 0 else horizontal_lines[i - 1]
            y_max = float('inf') if i == num_horizontal else horizontal_lines[i]

            for j in range(num_vertical + 1):
                x_min = 0 if j == 0 else vertical_lines[j - 1]
                x_max = float('inf') if j == num_vertical else vertical_lines[j]

                derivation_points = [p for p in all_points if x_min <= p[0] <= x_max and y_min <= p[1] <= y_max]
                if derivation_points:
                    derivation_name = f'Derivação (Linha {i + 1}, Coluna {j + 1})'
                    analysis_result = ecg_analyzer.analyze_ecg(derivation_points) # Chama a análise com os pontos segmentados
                    segmented_derivations[derivation_name] = analysis_result

        return jsonify({'results': segmented_derivations})

    except Exception as e:
        return jsonify({'error': f'Erro no processamento da análise segmentada: {e}'})

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True)