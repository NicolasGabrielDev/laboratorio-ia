import zipfile
import xml.etree.ElementTree as ET
import json
import os

docx_path = 'RBF2.docx'
ns = {'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}

def parse_table(table):
    rows = table.findall('w:tr', ns)
    result = []
    for row in rows:
        cells = row.findall('w:tc', ns)
        row_data = []
        for cell in cells:
            cell_text = ''.join(t.text or '' for t in cell.findall('.//w:t', ns))
            row_data.append(cell_text.strip())
        result.append(row_data)
    return result

if not os.path.exists(docx_path):
    print(f"Error: {docx_path} not found!")
    exit(1)

with zipfile.ZipFile(docx_path) as z:
    xml_content = z.read('word/document.xml')

root = ET.fromstring(xml_content)
body = root.find('.//w:body', ns)
tables = body.findall('w:tbl', ns)

print(f"Encontradas {len(tables)} tabelas no documento.")

# Vamos imprimir informações básicas sobre as tabelas para localizá-las
for i, table in enumerate(tables):
    rows = table.findall('w:tr', ns)
    if rows:
        cells = rows[0].findall('w:tc', ns)
        print(f"Tabela {i}: {len(rows)} linhas x {len(cells)} colunas. Exemplo célula (0,0): '{''.join(t.text or '' for t in cells[0].findall('.//w:t', ns)).strip()}'")

# De acordo com a nossa leitura do texto extraído:
# A tabela de testes (Validação) possui 15 amostras e está localizada em uma das primeiras tabelas.
# A tabela de treinamento possui 150 amostras dispostas em colunas paralelas e deve ser a última (ou uma das últimas).

# Vamos inspecionar e extrair de forma robusta
# Encontrando a tabela de teste: ela deve ter cabeçalho contendo "Amostra" ou "x1" ou "d" e cerca de 16-17 linhas (1 cabeçalho + 15 amostras + erro/variância)
tabela_teste_raw = None
tabela_treino_raw = None

for i, table in enumerate(tables):
    data = parse_table(table)
    if not data:
        continue
    # Verifica se é a tabela de validação
    # Geralmente contém "Amostra" no primeiro elemento da linha 0 e tem tamanho adequado
    if len(data) >= 16 and any("Amostra" in cell for cell in data[0]) and any("Erro Relativo" in cell for cell in data[-1]):
        tabela_teste_raw = data
        print(f"-> Identificada Tabela {i} como Tabela de Testes (Validação)")
    # A tabela de treinamento deve ter 150 amostras dispostas em colunas.
    # Ex: 50 linhas de dados, cada linha contendo 3 amostras paralelas (15 colunas no total)
    elif len(data) >= 50 and len(data[0]) >= 15:
        tabela_treino_raw = data
        print(f"-> Identificada Tabela {i} como Tabela de Treinamento")

# Caso a heurística automática precise de um ajuste, usamos os índices padrão
if tabela_teste_raw is None:
    # Heurística alternativa: a de teste tem ~18 linhas, a de treino tem ~51 linhas
    for i, table in enumerate(tables):
        data = parse_table(table)
        if 16 <= len(data) <= 25 and len(data[0]) >= 5:
            tabela_teste_raw = data
            print(f"-> Backup: Identificada Tabela {i} como Tabela de Testes")
            break

if tabela_treino_raw is None:
    for i, table in enumerate(tables):
        data = parse_table(table)
        if len(data) >= 50 and len(data[0]) >= 10:
            tabela_treino_raw = data
            print(f"-> Backup: Identificada Tabela {i} como Tabela de Treinamento")
            break

# ============================================================
# 1. PARSAR DADOS DE TREINAMENTO (150 amostras)
# ============================================================
training_data = []
if tabela_treino_raw:
    # Cabeçalho está na linha 0 (ex: Amostra, x1, x2, x3, d, Amostra, ...)
    # Os dados reais começam a partir da linha 1
    # Cada linha de dados tem 3 amostras lado a lado (offset 0, 5, 10)
    for r_idx, row in enumerate(tabela_treino_raw[1:], start=1):
        for offset in [0, 5, 10]:
            try:
                if offset >= len(row):
                    continue
                idx_str = row[offset].strip()
                if not idx_str:
                    continue
                
                idx = int(idx_str)
                x1 = float(row[offset+1].replace(',', '.'))
                x2 = float(row[offset+2].replace(',', '.'))
                x3 = float(row[offset+3].replace(',', '.'))
                d = float(row[offset+4].replace(',', '.'))
                
                training_data.append({
                    "amostra": idx,
                    "x1": x1,
                    "x2": x2,
                    "x3": x3,
                    "d": d
                })
            except (ValueError, IndexError) as e:
                # Evita crash em linhas em branco ou rodapés
                pass

    training_data.sort(key=lambda r: r['amostra'])

# ============================================================
# 2. PARSAR DADOS DE TESTE (15 amostras)
# ============================================================
test_data = []
if tabela_teste_raw:
    # A tabela de teste tem:
    # Linha 0: Título principal
    # Linha 1: Redes e colunas principais
    # Linhas 2 a 16: Amostras de 1 a 15 (índices 2:17)
    # Linhas seguintes: Erros relativos médios e variância
    # As colunas dos dados originais são as 5 primeiras: Amostra, x1, x2, x3, d
    for r_idx, row in enumerate(tabela_teste_raw[2:17], start=2):
        try:
            idx = int(row[0].strip())
            x1 = float(row[1].replace(',', '.'))
            x2 = float(row[2].replace(',', '.'))
            x3 = float(row[3].replace(',', '.'))
            d = float(row[4].replace(',', '.'))
            
            test_data.append({
                "amostra": idx,
                "x1": x1,
                "x2": x2,
                "x3": x3,
                "d": d
            })
        except (ValueError, IndexError) as e:
            print(f"Erro ao parsar linha {r_idx} de teste: {row}, erro: {e}")
            pass

# ============================================================
# 3. SALVAR E GERAR METADADOS
# ============================================================
with open('training_data.json', 'w', encoding='utf-8') as f:
    json.dump(training_data, f, indent=2, ensure_ascii=False)
print(f"Salvo: training_data.json ({len(training_data)} amostras)")

with open('test_data.json', 'w', encoding='utf-8') as f:
    json.dump(test_data, f, indent=2, ensure_ascii=False)
print(f"Salvo: test_data.json ({len(test_data)} amostras)")

metadata = {
    "problema": "Rede RBF - Sistema de Injeção Eletrônica de Combustível",
    "entradas": ["x1", "x2", "x3"],
    "saida": "y (quantidade de gasolina a ser injetada)",
    "topologias": [
        {"nome": "Rede 1", "N1": 5},
        {"nome": "Rede 2", "N1": 10},
        {"nome": "Rede 3", "N1": 15}
    ],
    "algoritmo_camada_oculta": "K-means",
    "algoritmo_camada_saida": "Regra Delta (Modo Batch)",
    "taxa_aprendizado": 0.01,
    "precisao": 1e-7,
    "num_treinamentos_por_topologia": 3,
    "pesos_iniciais": "aleatorios entre 0 e 1"
}

with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2, ensure_ascii=False)
print("Salvo: metadata.json")
