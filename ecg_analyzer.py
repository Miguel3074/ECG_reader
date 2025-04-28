import numpy as np
import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import base64
from io import BytesIO
import logging
import os
import math

# Configuração do logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Dicionário com condições do ECG (mantido como estava)
ecg_conditions = {
    "taquicardia": {
        "description":
        "A taquicardia é uma condição onde o coração bate mais rápido que o normal, geralmente acima de 100 batimentos por minuto.",
        "possible_causes":
        "Pode ser causada por febre, anemia, hipertireoidismo, estresse, exercício, entre outros.",
        "suggestions":
        "Considere avaliação médica para investigar a causa. Exames adicionais podem ser necessários."
    },
    "bradicardia": {
        "description":
        "A bradicardia é uma condição onde o coração bate mais devagar que o normal, geralmente abaixo de 60 batimentos por minuto.",
        "possible_causes":
        "Pode ser normal em atletas, mas também causada por doenças cardíacas, medicamentos, hipotireoidismo.",
        "suggestions":
        "Se sintomático (tontura, cansaço), procure avaliação médica. Pode ser necessário ajuste de medicação ou marcapasso."
    },
    "fibrilacao_atrial": {
        "description":
        "Fibrilação atrial é um tipo de arritmia onde os átrios do coração batem de forma caótica e irregular, resultando em um ritmo ventricular irregular.",
        "possible_causes":
        "Pode ser causada por doenças cardíacas, hipertensão, problemas da tireoide, idade avançada, consumo de álcool.",
        "suggestions":
        "Requer avaliação médica urgente. O tratamento visa controlar a frequência cardíaca, prevenir coágulos (risco de AVC) e restaurar o ritmo normal, se possível."
    },
    "normal": {
        "description":
        "O ritmo sinusal normal indica que a atividade elétrica do coração se origina no nó sinusal e a frequência cardíaca está entre 60 e 100 bpm, com ritmo regular.",
        "possible_causes":
        "Coração saudável em repouso.",
        "suggestions":
        "Continue monitorando sua saúde com exames regulares e mantenha um estilo de vida saudável."
    },
    "indeterminado": {
        "description":
        "A análise do ritmo ou frequência é indeterminada devido à qualidade do sinal ou dados insuficientes.",
        "possible_causes":
        "Sinal de ECG ruidoso, artefatos, extração inadequada da forma de onda, poucos batimentos detectados.",
        "suggestions":
        "Tente obter um traçado de ECG de melhor qualidade ou verifique o processo de extração da imagem."
    }
}


def preprocess_image(image_path):
    """ Carrega, converte para escala de cinza, aplica blur e threshold adaptativo """
    image = cv2.imread(image_path)
    if image is None:
        logging.error(f"Falha ao carregar a imagem: {image_path}")
        raise ValueError(
            "Falha ao carregar a imagem. Verifique o caminho do arquivo.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    try:
        binary = cv2.adaptiveThreshold(blurred, 255,
                                       cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY_INV, 11, 3)
    except Exception as e:
        logging.error(f"Erro durante adaptiveThreshold: {e}")
        _, binary = cv2.threshold(blurred, 0, 255,
                                  cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return binary


def detect_ecg_grid(gray_image):
    """Detecta as linhas da grade do ECG."""
    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges,
                            1,
                            np.pi / 180,
                            threshold=100,
                            minLineLength=50,
                            maxLineGap=10)
    horizontal_lines = []
    vertical_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if abs(y2 - y1) < 10:
                horizontal_lines.append(((x1, y1), (x2, y2)))
            elif abs(x2 - x1) < 10:
                vertical_lines.append(((x1, y1), (x2, y2)))
    return horizontal_lines, vertical_lines


def segment_ecg_leads(gray_image, horizontal_lines, vertical_lines):
    """Segmenta as doze derivações do ECG."""
    lead_rois = []
    h_lines_y = sorted(list(
        set([coord[1] for (x1, coord), (x2, _) in horizontal_lines])),
                       key=int)
    v_lines_x = sorted(list(
        set([coord[0] for (coord), (_, coord2) in vertical_lines] +
            [coord2[0] for (_), (coord, coord2) in vertical_lines])),
                       key=int)

    def crop(image, y1, y2, x1, x2):
        return image[int(y1):int(y2), int(x1):int(x2)]

    # Lógica de segmentação baseada na estrutura comum de ECG de 12 derivações
    # Assumindo 3 linhas de derivações, com 4 derivações cada (duas curtas, uma longa)
    # E uma tira de ritmo longa na parte inferior (que já estamos analisando separadamente)

    # Esta é uma lógica simplificada e pode precisar de ajustes finos
    if len(h_lines_y) >= 4 and len(v_lines_x) >= 3:
        try:
            lead_height_short = h_lines_y[1] - h_lines_y[0]
            lead_height_long = h_lines_y[3] - h_lines_y[
                2]  # Assumindo a última linha é mais longa
            lead_width_short = (v_lines_x[1] - v_lines_x[0]) if len(
                v_lines_x) > 1 else gray_image.shape[1] // 2
            lead_width_long = gray_image.shape[
                1]  # Largura total para as longas

            # Primeira linha: I, aVR
            lead_rois.append(
                crop(gray_image, h_lines_y[0], h_lines_y[1], v_lines_x[0],
                     v_lines_x[1]))  # I
            lead_rois.append(
                crop(gray_image, h_lines_y[0], h_lines_y[1], v_lines_x[1],
                     v_lines_x[-1]))  # aVR

            # Segunda linha: II, aVL
            lead_rois.append(
                crop(gray_image, h_lines_y[1], h_lines_y[2], v_lines_x[0],
                     v_lines_x[1]))  # II
            lead_rois.append(
                crop(gray_image, h_lines_y[1], h_lines_y[2], v_lines_x[1],
                     v_lines_x[-1]))  # aVL

            # Terceira linha: III, aVF
            lead_rois.append(
                crop(gray_image, h_lines_y[2], h_lines_y[3], v_lines_x[0],
                     v_lines_x[1]))  # III
            lead_rois.append(
                crop(gray_image, h_lines_y[2], h_lines_y[3], v_lines_x[1],
                     v_lines_x[-1]))  # aVF

            # Quarta linha (assumindo precordiais mais longas): V1, V2, V3, V4, V5, V6
            # Isso é uma SUPOSIÇÃO FORTE sobre o layout. Ajuste conforme necessário.
            num_v_lines = len(v_lines_x)
            if num_v_lines >= 7 and len(h_lines_y) >= 5:
                v_width = (v_lines_x[-1] - v_lines_x[0]) / 6
                for i in range(6):
                    start_x = v_lines_x[0] + int(i * v_width)
                    end_x = v_lines_x[0] + int((i + 1) * v_width)
                    lead_rois.append(
                        crop(gray_image, h_lines_y[3], h_lines_y[4], start_x,
                             end_x))
            elif len(
                    h_lines_y
            ) >= 7:  # Tentativa alternativa se houver mais linhas horizontais
                lead_height_precordial = h_lines_y[4] - h_lines_y[3]
                for i in range(
                        6
                ):  # Assumindo 6 derivações precordiais em duas linhas
                    row = i // 3
                    col = i % 3
                    start_y = h_lines_y[3 + row]
                    end_y = h_lines_y[4 + row]
                    start_x = v_lines_x[0] + int(col * lead_width_short)
                    end_x = v_lines_x[0] + int((col + 1) * lead_width_short)
                    lead_rois.append(
                        crop(gray_image, start_y, end_y, start_x, end_x))

        except IndexError:
            logging.warning(
                "Não foi possível segmentar todas as 12 derivações devido a detecção de grade incompleta."
            )
            return []

    return lead_rois


def extract_waveform_from_roi_centroid(roi):
    """ Extrai a forma de onda de uma ROI usando o centróide dos pixels brancos por coluna """
    height_roi, width_roi = roi.shape
    if width_roi == 0 or height_roi == 0:
        logging.warning("ROI com dimensões zero.")
        return None

    waveform_roi = np.zeros(width_roi)
    vertical_margin = int(0.1 * height_roi)
    vertical_start = vertical_margin
    vertical_end = height_roi - vertical_margin

    last_valid_y = (vertical_start + vertical_end) / 2

    for x in range(width_roi):
        column = roi[vertical_start:vertical_end, x]
        white_pixels_indices = np.where(column > 0)[0]

        if white_pixels_indices.size > 0:
            centroid_relative_y = np.mean(white_pixels_indices)
            current_y = vertical_start + centroid_relative_y
            waveform_roi[x] = current_y
            last_valid_y = current_y
        else:
            waveform_roi[x] = last_valid_y

    min_val = np.min(waveform_roi)
    max_val = np.max(waveform_roi)
    if max_val > min_val:
        waveform_roi = (waveform_roi - min_val) / (max_val - min_val)
    else:
        logging.warning(
            f"Sinal constante ou inválido extraído da ROI (min={min_val}, max={max_val})."
        )
        return None

    window_length = 15
    polyorder = 3
    if len(waveform_roi) > window_length:
        try:
            smoothed_waveform = signal.savgol_filter(waveform_roi,
                                                     window_length,
                                                     polyorder,
                                                     mode='nearest')
            return smoothed_waveform
        except Exception as e:
            logging.error(
                f"Erro no filtro Savitzky-Golay: {e}. Retornando forma de onda não filtrada."
            )
            return waveform_roi
    else:
        logging.warning(
            f"Forma de onda muito curta ({len(waveform_roi)} pontos) para suavização com janela {window_length}. Retornando não filtrada."
        )
        return waveform_roi


def detect_r_peaks(waveform, sample_rate_hz):
    """ Detecta picos R usando um método baseado em derivada e threshold adaptativo """
    if waveform is None or len(waveform) < 5 or sample_rate_hz <= 0:
        logging.warning(
            "Dados de forma de onda insuficientes ou sample_rate inválido para detecção de picos R."
        )
        return []

    try:
        filtered_waveform = signal.savgol_filter(waveform,
                                                 11,
                                                 3,
                                                 mode='nearest')
    except ValueError:
        filtered_waveform = waveform

    derivative = np.diff(filtered_waveform)
    derivative = np.append(derivative, derivative[-1])
    squared_derivative = derivative**2
    window_size_ms = 150
    window_size_samples = int(window_size_ms / 1000 * sample_rate_hz)
    if window_size_samples < 1: window_size_samples = 1
    if window_size_samples > len(squared_derivative):
        window_size_samples = len(squared_derivative)
    integrated_signal = np.convolve(squared_derivative,
                                    np.ones(window_size_samples) /
                                    window_size_samples,
                                    mode='same')

    min_distance_ms = 240
    min_distance_samples = int(min_distance_ms / 1000 * sample_rate_hz)
    if min_distance_samples < 1: min_distance_samples = 1

    signal_peak = np.max(integrated_signal)
    noise_level = np.mean(integrated_signal[:int(sample_rate_hz)])
    threshold = noise_level + 0.5 * (signal_peak - noise_level)
    threshold = max(float(threshold), float(0.1 * signal_peak),
                    float(np.finfo(float).eps))

    try:
        peaks, _ = signal.find_peaks(integrated_signal,
                                     height=threshold,
                                     distance=min_distance_samples)
        logging.info(
            f"Detectados {len(peaks)} picos R com sample_rate={sample_rate_hz:.2f} Hz, threshold={threshold:.4f}, min_distance={min_distance_samples} samples."
        )
        return peaks
    except Exception as e:
        logging.error(f"Erro durante find_peaks: {e}")
        return []


def analyze_waveform(waveform, sample_rate_hz):
    """ Analisa a forma de onda para calcular FC e ritmo """
    results = {
        'heart_rate': None,
        'rhythm_analysis': 'Indeterminado',
        'rr_intervals_ms': []
    }

    if waveform is None or sample_rate_hz <= 0:
        results['rhythm_analysis'] = "Forma de onda inválida ou ausente"
        return results

    r_peaks = detect_r_peaks(waveform, sample_rate_hz)

    if len(r_peaks) < 2:
        logging.warning(
            f"Picos R insuficientes detectados ({len(r_peaks)}) para análise de ritmo."
        )
        results['rhythm_analysis'] = "Picos R insuficientes detectados"
        return results

    rr_intervals_samples = np.diff(r_peaks)
    rr_intervals_sec = rr_intervals_samples / sample_rate_hz
    results['rr_intervals_ms'] = (rr_intervals_sec * 1000).tolist()

    if len(rr_intervals_sec) == 0:
        results['rhythm_analysis'] = "Não foi possível calcular intervalos RR"
        return results

    avg_rr_sec = np.mean(rr_intervals_sec)
    std_rr_sec = np.std(rr_intervals_sec)

    if avg_rr_sec > 0:
        heart_rate = 60 / avg_rr_sec
        results['heart_rate'] = round(heart_rate)
    else:
        results['heart_rate'] = None
        results['rhythm_analysis'] = "Intervalo RR médio inválido"
        return results

    if avg_rr_sec > 0:
        cv_rr = std_rr_sec / avg_rr_sec
    else:
        cv_rr = float('inf')

    std_rr_ms = std_rr_sec * 1000
    logging.info(
        f"Análise: Média RR={avg_rr_sec*1000:.1f} ms, STD RR={std_rr_ms:.1f} ms, CV={cv_rr*100:.1f}%"
    )

    if std_rr_ms < 80 and cv_rr < 0.12:
        rhythm = "Ritmo Regular"
    elif std_rr_ms > 120 or cv_rr > 0.18:
        rhythm = "Ritmo Irregular"
    else:
        rhythm = "Ligeira Irregularidade / Arritmia Sinusal"

    results['rhythm_analysis'] = rhythm

    return results


def generate_visualization(waveform, analysis_results, sample_rate_hz):
    """ Gera uma visualização da forma de onda com picos R marcados """
    if waveform is None or len(waveform) == 0 or sample_rate_hz <= 0:
        return None

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, ax = plt.subplots(figsize=(12, 4))

    time_axis = np.arange(len(waveform)) / sample_rate_hz

    ax.plot(time_axis,
            waveform,
            color='cyan',
            linewidth=1.0,
            label='ECG Waveform')

    r_peaks_indices = detect_r_peaks(waveform, sample_rate_hz)
    if len(r_peaks_indices) > 0:
        r_peaks_time = r_peaks_indices / sample_rate_hz
        r_peaks_amplitude = waveform[r_peaks_indices]
        ax.plot(r_peaks_time,
                r_peaks_amplitude,
                'ro',
                markersize=5,
                label='Detected R-peaks')

    ax.set_xlabel('Tempo (s)')
    ax.set_ylabel('Amplitude Normalizada')
    hr = analysis_results.get('heart_rate', 'N/A')
    rhythm = analysis_results.get('rhythm_analysis', 'N/A')
    ax.set_title(f"Forma de Onda ECG Analisada\nFC: {hr} BPM, Ritmo: {rhythm}")
    ax.legend()
    ax.grid(True, linestyle=':', alpha=0.7)

    buffer = BytesIO()
    try:
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close(fig)
        return img_base64
    except Exception as e:
        logging.error(f"Erro ao gerar visualização: {e}")
        plt.close(fig)
        return None


def provide_explanation(rhythm_analysis, heart_rate):
    """ Fornece explicações baseadas no ritmo e frequência cardíaca """
    condition = "indeterminado"

    if rhythm_analysis is None or heart_rate is None or heart_rate == 'Não detectado':
        condition = "indeterminado"
    elif "irregular" in rhythm_analysis.lower():
        if "Ritmo Irregular" in rhythm_analysis and (heart_rate > 90
                                                     or heart_rate < 50):
            condition = "fibrilacao_atrial"
        else:
            condition = "indeterminado"
    elif "regular" in rhythm_analysis.lower():
        if heart_rate > 100:
            condition = "taquicardia"
        elif heart_rate < 60:
            condition = "bradicardia"
        else:
            condition = "normal"

    if condition == "indeterminado" and heart_rate is not None and heart_rate != 'Não detectado':
        if heart_rate > 100:
            condition = "taquicardia"
        elif heart_rate < 60:
            condition = "bradicardia"

    return ecg_conditions.get(condition, ecg_conditions["indeterminado"])


def analyze_single_lead(lead_image, sample_rate_hz, lead_name=""):
    """ Analisa uma única derivação do ECG """
    logging.info(f"Analisando derivação: {lead_name}")
    waveform = extract_waveform_from_roi_centroid(lead_image)
    if waveform is None:
        logging.warning(
            f"Falha ao extrair forma de onda da derivação {lead_name}")
        return {
            'heart_rate': None,
            'rhythm_analysis': 'Não detectado',
            'visualization': None
        }

    analysis = analyze_waveform(waveform, sample_rate_hz)
    visualization = generate_visualization(waveform, analysis, sample_rate_hz)
    return {
        'heart_rate': analysis.get('heart_rate'),
        'rhythm_analysis': analysis.get('rhythm_analysis'),
        'visualization': visualization
    }


def analyze_ecg_twelve_leads(image_path, sample_rate_hz):
    """Função principal para analisar um ECG de doze derivações."""
    try:
        logging.info(
            f"Iniciando análise de 12 derivações para {image_path} com sample_rate={sample_rate_hz} Hz"
        )
        preprocessed_img = preprocess_image(image_path)
        debug_folder = "debug_ecg_analysis_12"
        os.makedirs(debug_folder, exist_ok=True)
        cv2.imwrite(os.path.join(debug_folder, "01_preprocessed.png"),
                    preprocessed_img)

        horizontal_grid, vertical_grid = detect_ecg_grid(preprocessed_img)
        grid_image = cv2.cvtColor(preprocessed_img, cv2.COLOR_GRAY2BGR)
        for (x1, y1), (x2, y2) in horizontal_grid:
            cv2.line(grid_image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        for (x1, y1), (x2, y2) in vertical_grid:
            cv2.line(grid_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.imwrite(os.path.join(debug_folder, "02_grid_detection.png"),
                    grid_image)

        lead_images = segment_ecg_leads(preprocessed_img, horizontal_grid,
                                        vertical_grid)
        if not lead_images or len(lead_images) < 12:
            logging.error(
                f"Falha na segmentação das 12 derivações. {len(lead_images)} derivações segmentadas."
            )
            return {
                'error': 'Falha na segmentação das derivações',
                'explanation': provide_explanation('indeterminado', None),
                'visualization': None,
                'lead_analyses': {}
            }

        lead_analyses = {}
        lead_names = [
            "I", "aVR", "II", "aVL", "III", "aVF", "V1", "V2", "V3", "V4",
            "V5", "V6"
        ]
        for i, lead_img in enumerate(lead_images):
            lead_name = lead_names[i] if i < len(lead_names) else f"Lead_{i+1}"
            cv2.imwrite(os.path.join(debug_folder, f"03_lead_{lead_name}.png"),
                        lead_img)
            analysis = analyze_single_lead(lead_img, sample_rate_hz, lead_name)
            lead_analyses[lead_name] = analysis

        # Integração dos resultados (foco na derivação II para ritmo e FC geral)
        overall_rhythm = lead_analyses.get("II",
                                           {}).get("rhythm_analysis",
                                                   "Indeterminado")
        overall_heart_rate = lead_analyses.get("II",
                                               {}).get("heart_rate",
                                                       "Não detectado")
        overall_visualization = lead_analyses.get("II",
                                                  {}).get("visualization")
        explanation = provide_explanation(overall_rhythm, overall_heart_rate)

        return {
            'heart_rate': overall_heart_rate,
            'rhythm_analysis': overall_rhythm,
            'explanation': explanation,
            'visualization': overall_visualization,
            'lead_analyses': lead_analyses
        }

    except Exception as e:
        logging.exception(
            f"Erro inesperado na análise do ECG de 12 derivações: {e}")
        return {
            'error': f"Erro inesperado: {e}",
            'explanation': provide_explanation('indeterminado', None),
            'visualization': None,
            'lead_analyses': {}
        }


def analyze_ecg(image_path, roi_coords, sample_rate_hz):
    """Função principal para analisar uma ROI de ECG (mantida para compatibilidade com código anterior)"""
    try:
        logging.info(
            f"Iniciando análise para ROI em {image_path} com sample_rate={sample_rate_hz} Hz, ROI={roi_coords}"
        )
        preprocessed_img = preprocess_image(image_path)
        debug_folder = "debug_ecg_analysis_roi"
        os.makedirs(debug_folder, exist_ok=True)
        cv2.imwrite(os.path.join(debug_folder, "01_preprocessed.png"),
                    preprocessed_img)

        y_start, y_end, x_start, x_end = roi_coords
        roi = preprocessed_img[y_start:y_end, x_start:x_end]
        logging.info(
            f"Analisando ROI (dimensões: {roi.shape[0]}x{roi.shape[1]})")
        cv2.imwrite(os.path.join(debug_folder, "02_selected_roi.png"), roi)

        waveform = extract_waveform_from_roi_centroid(roi)

        if waveform is None:
            logging.error("Falha ao extrair a forma de onda da ROI.")
            return {
                'heart_rate': 'Não detectado',
                'rhythm_analysis': 'Falha na extração da forma de onda',
                'visualization': None,
                'explanation': provide_explanation('indeterminado', None),
                'error': 'Waveform extraction failed'
            }

        try:
            np.savetxt(os.path.join(debug_folder, "03_extracted_waveform.csv"),
                       waveform,
                       delimiter=",")
            logging.info(f"Forma de onda extraída salva em {debug_folder}")
        except Exception as e_save:
            logging.warning(
                f"Não foi possível salvar a forma de onda em CSV: {e_save}")

        analysis = analyze_waveform(waveform, sample_rate_hz)
        logging.info(f"Resultados da análise: {analysis}")

        visualization = generate_visualization(waveform, analysis,
                                               sample_rate_hz)
        explanation = provide_explanation(analysis['rhythm_analysis'],
                                          analysis['heart_rate'])

        return {
            'heart_rate':
            analysis['heart_rate']
            if analysis['heart_rate'] is not None else 'Não detectado',
            'rhythm_analysis':
            analysis['rhythm_analysis'],
            'rr_intervals_ms':
            analysis.get('rr_intervals_ms', []),
            'visualization':
            visualization,
            'explanation':
            explanation,
            'error':
            None
        }

    except ValueError as ve:
        logging.error(f"Erro de valor durante análise: {str(ve)}")
        return {
            'error': str(ve),
            'visualization': None,
            'explanation': provide_explanation('indeterminado', None)
        }
    except Exception as e:
        logging.exception(f"Erro inesperado na análise do ECG: {str(e)}")
        return {
            'heart_rate': None,
            'rhythm_analysis': None,
            'visualization': None,
            'explanation': provide_explanation('indeterminado', None),
            'error': f"Erro inesperado: {str(e)}"
        }

        # --- Exemplo de Uso ---


if __name__ == "__main__":
    image_file = 'ecg.jpg'

    # Taxa de amostragem estimada (pixels por segundo) - PRECISA SER CALCULADA/MEDIDA!
    estimated_sample_rate = 95.0  # Hz (pixels/segundo) - AJUSTE ESTE VALOR!

    if not os.path.exists(image_file):
        print(f"Erro: Arquivo de imagem não encontrado: {image_file}")
    else:
        # Análise do ECG de 12 Derivações
        analysis_result_12_leads = analyze_ecg_twelve_leads(
            image_file, estimated_sample_rate)
        print("\n--- Resultados da Análise do ECG de 12 Derivações ---")
        if analysis_result_12_leads.get('error'):
            print(
                f"Erro durante a análise: {analysis_result_12_leads['error']}")
        else:
            print(
                f"Frequência Cardíaca (FC): {analysis_result_12_leads.get('heart_rate', 'N/A')} BPM"
            )
            print(
                f"Análise do Ritmo: {analysis_result_12_leads.get('rhythm_analysis', 'N/A')}"
            )
            explanation = analysis_result_12_leads.get('explanation', {})
            print("\n--- Explicação ---")
            print(f"  Descrição: {explanation.get('description', 'N/A')}")
            print(
                f"  Causas Possíveis: {explanation.get('possible_causes', 'N/A')}"
            )
            print(f"  Sugestões: {explanation.get('suggestions', 'N/A')}")

            if analysis_result_12_leads.get('visualization'):
                html_content = f"""
                                <html><head><title>Visualização ECG (Derivação II)</title></head>
                                <body style="background-color: #f0f0f0; text-align: center;">
                                <h2>Visualização da Forma de Onda ECG (Derivação II)</h2>
                                <img src="data:image/png;base64,{analysis_result_12_leads['visualization']}" alt="ECG Waveform Visualization" style="max-width: 90%; border: 1px solid #ccc;">
                                <h3>Análise por Derivação:</h3>
                                <ul>
                                {''.join([f'<li><b>{lead}:</b> FC: {data.get("heart_rate", "N/A")}, Ritmo: {data.get("rhythm_analysis", "N/A")}</li>' for lead, data in analysis_result_12_leads.get("lead_analyses", {}).items()])}
                                </ul>
                                </body></html>
                                """
                output_html = "ecg_visualization_12_leads.html"
                with open(output_html, "w") as f:
                    f.write(html_content)
                print(
                    f"\nVisualização das 12 derivações (com foco na II) salva em: {output_html}"
                )
            else:
                print("\nVisualização da Derivação II não pôde ser gerada.")

        # Análise da ROI única (tira de ritmo II) para manter compatibilidade
        roi_lead_ii_bottom = (645, 715, 10, 1010)  # Ajuste conforme necessário
        analysis_result_roi = analyze_ecg(image_file, roi_lead_ii_bottom,
                                          estimated_sample_rate)
        print("\n--- Resultados da Análise da ROI (Derivação II Inferior) ---")
        if analysis_result_roi.get('error'):
            print(
                f"Erro durante a análise da ROI: {analysis_result_roi['error']}"
            )
        else:
            print(
                f"Frequência Cardíaca (ROI): {analysis_result_roi.get('heart_rate', 'N/A')} BPM"
            )
            print(
                f"Análise do Ritmo (ROI): {analysis_result_roi.get('rhythm_analysis', 'N/A')}"
            )
