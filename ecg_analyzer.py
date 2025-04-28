import csv
import numpy as np
from collections import defaultdict

def analyze_ecg(csv_filepath):
    """
    Analisa o ECG, segmentando as 12 derivações padrão e a tira de ritmo usando estrutura em grade.
    """
    print(f"Analisando ECG: {csv_filepath} (segmentação por grade)...")
    try:
        all_points = []
        with open(csv_filepath, 'r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            header = next(reader)
            for row in reader:
                try:
                    x = int(row[0])
                    y = int(row[1])
                    all_points.append((x, y))
                except ValueError:
                    print(f"Erro ao converter linha para inteiro: {row}")
        if not all_points:
            return {"erro": "Nenhum dado de ECG encontrado."}

        all_points.sort(key=lambda p: p[1])

        y_grouped = defaultdict(list)
        y_tolerance_initial = 15
        for x, y in all_points:
            found_group = False
            for avg_y in list(y_grouped.keys()):
                if abs(y - avg_y) <= y_tolerance_initial:
                    y_grouped[avg_y].append((x, y))
                    found_group = True
                    break
            if not found_group:
                y_grouped[y].append((x, y))

        lines = sorted([(np.mean([p[1] for p in points]), points) for avg_y, points in y_grouped.items()], key=lambda item: item[0])

        derivations = {}
        derivation_names = ["I", "aVR", "V1", "V4", "II", "aVL", "V2", "V5", "III", "aVF", "V3", "V6"]
        derivation_index = 0

        main_derivation_data = []
        main_derivation_name = "MAIN"
        main_y_mean = -1

        if len(lines) >= 3:
            for i in range(min(3, len(lines))):
                avg_y, points = lines[i]
                points.sort(key=lambda p: p[0])
                if not points:
                    continue

                min_x = min(p[0] for p in points)
                max_x = max(p[0] for p in points)
                if max_x > min_x:
                    width = max_x - min_x
                    segment_width = width / 4
                    for j in range(4):
                        start_x = min_x + j * segment_width
                        end_x = min_x + (j + 1) * segment_width
                        segment_points = [p for p in points if start_x <= p[0] < end_x]
                        if derivation_index < len(derivation_names) and len(segment_points) > 50:
                            derivations[f"{derivation_names[derivation_index]} (Y médio: {int(avg_y)})"] = segment_points
                            derivation_index += 1

            # Identificar a possível tira de ritmo (a linha mais longa em X ou com Y diferente)
            if len(lines) > 3:
                longest_line_x = 0
                longest_line_points = []
                longest_line_y_mean = -1
                for avg_y, points in lines:
                    if not points:
                        continue
                    x_coords = [p[0] for p in points]
                    current_length = max(x_coords) - min(x_coords)
                    if current_length > longest_line_x and len(points) > 200:
                        longest_line_x = current_length
                        longest_line_points = points
                        longest_line_y_mean = avg_y
                    elif len(lines) > 3 and abs(avg_y - np.mean([l[0] for l in lines[:3]])) > y_tolerance_initial * 2 and len(points) > 200 and current_length > longest_line_x / 2:
                        longest_line_x = current_length
                        longest_line_points = points
                        longest_line_y_mean = avg_y


                if longest_line_points:
                    main_derivation_data = longest_line_points
                    main_y_mean = longest_line_y_mean


        analysis_results = {}
        def find_r_peaks(data, prominence=5):
            if not data: return []
            y_values = np.array([point[1] for point in data])
            x_values = np.array([point[0] for point in data])
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(-y_values, prominence=prominence)
            return x_values[peaks]

        for derivation_name, derivation_data in derivations.items():
            analysis = {}
            r_peak_x_indices = find_r_peaks(derivation_data, prominence=15)
            if len(r_peak_x_indices) >= 2:
                distances = np.diff(r_peak_x_indices)
                if distances.size > 0:
                    analysis["frequencia_cardiaca"] = f"Média de distância R-R: {np.mean(distances):.2f}"
                else:
                    analysis["frequencia_cardiaca"] = "Poucos picos R detectados."
            else:
                analysis["frequencia_cardiaca"] = "Não detectou picos R suficientes."
            analysis["ritmo_sinusal"] = "Requer análise P e QRS."
            analysis_results[derivation_name] = analysis

        main_analysis = {}
        if main_derivation_data:
            r_peak_x_indices_main = find_r_peaks(main_derivation_data, prominence=15)
            if len(r_peak_x_indices_main) >= 2:
                distances_main = np.diff(r_peak_x_indices_main)
                if distances_main.size > 0:
                    main_analysis["frequencia_cardiaca"] = f"Média de distância R-R: {np.mean(distances_main):.2f}"
                else:
                    main_analysis["frequencia_cardiaca"] = "Poucos picos R detectados (MAIN)."
            else:
                main_analysis["frequencia_cardiaca"] = "Não detectou picos R suficientes (MAIN)."
            main_analysis["ritmo_sinusal"] = "Requer análise P e QRS (MAIN)."

        print(f"Análise concluída (segmentou em {len(derivations)} derivações e 'MAIN').")
        return {"derivations": analysis_results, "main": main_analysis}

    except FileNotFoundError:
        return {"erro": f"Arquivo não encontrado: {csv_filepath}"}
    except Exception as e:
        return {"erro": f"Erro na análise: {e}"}

if __name__ == '__main__':
    csv_file = 'static/uploads/ecg_image_data.csv'
    results = analyze_ecg(csv_file)
    print("\nResultados da Análise do ECG:")
    if "derivations" in results:
        for derivation, analysis in results["derivations"].items():
            print(f"\n--- {derivation} ---")
            for key, value in analysis.items():
                print(f"{key}: {value}")
    if "main" in results and results["main"]:
        print("\n--- Derivação MAIN ---")
        for key, value in results["main"].items():
            print(f"{key}: {value}")
    elif "main" in results:
        print("\n--- Derivação MAIN ---")
        print("Nenhuma análise disponível.")