import zipfile
import xml.etree.ElementTree as ET
import json

docx_path = 'PMC1.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

with zipfile.ZipFile(docx_path) as z:
    xml_content = z.read('word/document.xml')

root = ET.fromstring(xml_content)
body = root.find('.//w:body', ns)
tables = body.findall('w:tbl', ns)

def parse_table(table):
    rows = table.findall('w:tr', ns)
    result = []
    for row in rows:
        cells = row.findall('w:tc', ns)
        result.append([''.join(t.text or '' for t in cell.findall('.//w:t', ns)) for cell in cells])
    return result

# Tabela 4 - Dados de TREINAMENTO (200 amostras, 3 colunas de amostras por linha)
train_raw = parse_table(tables[3])
training_data = []
header = train_raw[0]  # ['Amostra','x1','x2','x3','d', 'Amostra','x1','x2','x3','d', ...]
for row in train_raw[1:]:
    for offset in [0, 5, 10]:
        try:
            idx = row[offset].strip()
            if idx == '':
                continue
            training_data.append({
                "amostra": int(idx),
                "x1": float(row[offset+1]),
                "x2": float(row[offset+2]),
                "x3": float(row[offset+3]),
                "d":  float(row[offset+4])
            })
        except (ValueError, IndexError):
            pass

training_data.sort(key=lambda r: r['amostra'])

with open('training_data.json', 'w') as f:
    json.dump(training_data, f, indent=2)
print(f"Dados de treinamento salvos: {len(training_data)} amostras -> training_data.json")

# Tabela 3 - Dados de TESTE (20 amostras)
test_raw = parse_table(tables[2])
test_data = []
for row in test_raw[1:21]:
    try:
        test_data.append({
            "amostra": int(row[0]),
            "x1": float(row[1]),
            "x2": float(row[2]),
            "x3": float(row[3]),
            "d":  float(row[4])
        })
    except (ValueError, IndexError):
        pass

with open('test_data.json', 'w') as f:
    json.dump(test_data, f, indent=2)
print(f"Dados de teste salvos: {len(test_data)} amostras -> test_data.json")

# Metadados do problema
metadata = {
    "problema": "Rede Perceptron Multicamadas - Sistema de Ressonância Magnética",
    "entradas": ["x1", "x2", "x3"],
    "saida": "y (energia absorvida)",
    "topologia": {"entradas": 3, "camada_oculta": 12, "camada_saida": 10, "saidas": 1},
    "algoritmo": "Backpropagation (Regra Delta Generalizada)",
    "funcao_ativacao": "logistica (sigmoid)",
    "taxa_aprendizado": 0.1,
    "precisao": 1e-6,
    "num_treinamentos": 5,
    "pesos_iniciais": "aleatorios entre 0 e 1"
}

with open('metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("Metadados salvos -> metadata.json")
