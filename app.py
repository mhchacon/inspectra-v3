from flask import Flask, request, render_template, send_file, Response
from werkzeug.utils import secure_filename
import io
from ultralytics import YOLO
import numpy as np
from PIL import Image
import cv2
import os
from datetime import datetime
import base64  
from flask_sqlalchemy import SQLAlchemy

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'

# Configuração do banco de dados SQLite
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///inspections.db'  # Arquivo SQLite no mesmo diretório
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False  # Evita avisos desnecessários

# Inicializa o banco de dados
db = SQLAlchemy(app)

# Certifique-se de que a pasta de uploads existe
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Função para criar o filtro b64encode
def b64encode_filter(data):
    if isinstance(data, str):
        data = data.encode('utf-8')
    return base64.b64encode(data).decode('utf-8')

# Registre o filtro no ambiente do Jinja2
app.jinja_env.filters['b64encode'] = b64encode_filter




class Detection:
    def __init__(self):
        # Baixe os pesos do YOLO e altere o caminho conforme necessário
        self.model = YOLO(r"yolov11_custom.pt")

    def predict(self, img, classes=[], conf=0.5):
        if classes:
            results = self.model.predict(img, classes=classes, conf=conf)
        else:
            results = self.model.predict(img, conf=conf)
        return results

    def predict_and_detect(self, img, classes=[], conf=0.5, rectangle_thickness=2, text_thickness=1):
        results = self.predict(img, classes, conf=conf)
        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]

                # Defina uma cor padrão para casos não previstos
                box_color = (255, 0, 0)  # Vermelho (padrão)

                if class_name == "Intacto":
                    box_color = (0, 255, 0)  # Verde para intactos
                elif class_name == "Danificado":
                    box_color = (0, 0, 255)  # Azul para danificados

                # Desenha o quadrado e o texto
                cv2.rectangle(img, (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                            (int(box.xyxy[0][2]), int(box.xyxy[0][3])), box_color, rectangle_thickness)
            
                cv2.putText(img, f"{class_name}",
                            (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                            cv2.FONT_HERSHEY_PLAIN, 1, box_color, text_thickness)
        return img, results

    def detect_from_image(self, image):
        result_img, _ = self.predict_and_detect(image, classes=[], conf=0.5)
        return result_img

    def detect_from_video_frame(self, frame):
        result_img, _ = self.predict_and_detect(frame, classes=[], conf=0.5)
        return result_img


detection = Detection()

class InspectionResult(db.Model):
    id = db.Column(db.Integer, primary_key=True)  # ID único
    filename = db.Column(db.String(100), nullable=False)  # Nome do arquivo da imagem
    intact_count = db.Column(db.Integer, nullable=False)  # Contagem de objetos intactos
    damaged_count = db.Column(db.Integer, nullable=False)  # Contagem de objetos danificados
    inspection_time = db.Column(db.DateTime, nullable=False)  # Horário da inspeção
    image_base64 = db.Column(db.Text, nullable=False)  # Imagem processada em base64

    def __repr__(self):
        return f"InspectionResult(id={self.id}, filename={self.filename}, intact_count={self.intact_count}, damaged_count={self.damaged_count})"


@app.route('/')
def index():
    return monitoramento()  

@app.route('/monitoramento')
def monitoramento():
    results = InspectionResult.query.all()

    # Contagem de intactos e danificados
    intact_count = sum(1 for result in results if result.damaged_count == 0)
    damaged_count = sum(1 for result in results if result.damaged_count > 0)

    # Garantir que os valores sejam números
    intact_count = intact_count if intact_count is not None else 0
    damaged_count = damaged_count if damaged_count is not None else 0

    return render_template('monitoramento.html',
                           results=results,
                           intact_count=intact_count,
                           damaged_count=damaged_count)


@app.route('/object-detection/', methods=['POST'])
def apply_detection():
    if 'image' not in request.files:
        return 'No file part', 400

    file = request.files['image']
    if file.filename == '':
        return 'No selected file', 400

    if file:
        try:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Processar a imagem (código existente)
            img = cv2.imread(file_path)
            img = cv2.resize(img, (512, 512))
            result_img, results = detection.predict_and_detect(img, classes=[], conf=0.5)

            # Contar objetos intactos e danificados
            intact_count = 0
            damaged_count = 0
            for result in results:
                for box in result.boxes:
                    class_id = int(box.cls[0])
                    class_name = result.names[class_id]
                    if class_name == "Intacto":
                        intact_count += 1
                    elif class_name == "Danificado":
                        damaged_count += 1

            # Registrar horário da inspeção
            inspection_time = datetime.now()

            # Converter imagem para base64
            result_img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
            output = Image.fromarray(result_img_rgb)
            buf = io.BytesIO()
            output.save(buf, format="PNG")
            buf.seek(0)
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')

            # Salvar no banco de dados
            new_inspection = InspectionResult(
                filename=filename,
                intact_count=intact_count,
                damaged_count=damaged_count,
                inspection_time=inspection_time,
                image_base64=image_base64
            )
            db.session.add(new_inspection)
            db.session.commit()

            # Remover arquivo original
            os.remove(file_path)

            # Renderizar template
            return render_template('index.html',
                                   intact_count=intact_count,
                                   damaged_count=damaged_count,
                                   inspection_time=inspection_time.strftime("%Y-%m-%d %H:%M:%S"),
                                   image_data=image_base64,
                                   message="Imagem processada com sucesso!",
                                   is_success=True)

        except Exception as e:
            # Log do erro e remoção do arquivo temporário
            print(f"Erro ao processar a imagem: {str(e)}")
            if 'file_path' in locals() and os.path.exists(file_path):
                os.remove(file_path)
            return f"Erro interno: {str(e)}", 500

@app.route('/results')
def show_results():
    results = InspectionResult.query.all()  # Busca todos os resultados
    return render_template('results.html', results=results)
        
@app.route('/video')
def index_video():
    return render_template('video.html')


def gen_frames():
    cap = cv2.VideoCapture(0)  # Use 0 para a câmera padrão
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Redimensiona o frame para 512x512
        frame = cv2.resize(frame, (512, 512))

        # Aplica a detecção de objetos no frame
        frame = detection.detect_from_video_frame(frame)

        # Converte o frame para JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Converte o buffer para bytes
        frame_bytes = buffer.tobytes()

        # Retorna o frame no formato de streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    # Libera a câmera quando o loop termina
    cap.release()


@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


# Rota para obter os dados do relatório (opcional)
@app.route('/get-report')
def get_report():
    # Simula a obtenção dos dados do relatório
    intact_count = 5  # Substitua pela lógica real
    damaged_count = 3  # Substitua pela lógica real
    inspection_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    return {
        'intact_count': intact_count,
        'damaged_count': damaged_count,
        'inspection_time': inspection_time
    }


if __name__ == '__main__':
    with app.app_context():
        db.create_all()  # Cria as tabelas no banco de dados
    app.run(host="0.0.0.0", port=8000, debug=True)