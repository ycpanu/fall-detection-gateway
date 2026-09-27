import json
import time
import os
import shutil
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, UploadFile, File, Request
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timedelta

# 1. 数据库配置 (SQLite)
SQLALCHEMY_DATABASE_URL = "sqlite:///./cloud_alerts.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 报警记录表
class AlertRecord(Base):
    __tablename__ = "alerts"

    id = Column(
        Integer,
        primary_key=True,
        index=True,
        autoincrement=True
    )

    # 网关生成的全局唯一事件 ID
    event_id = Column(
        String,
        unique=True,
        index=True,
        nullable=False
    )

    device_id = Column(
        String,
        index=True,
        nullable=False,
    )

    deployment_area = Column(
        String,
        default = "未配置区域"
    )

    # FALL / HELP_REQUEST
    event_type = Column(
        String,
        nullable=False
    )

    # JSON字符串，例如：
    # ["VISION"]
    # ["VOICE"]
    # ["VISION", "VOICE"]
    sources = Column(
        String,
        default="[]"
    )

    timestamp = Column(Integer)

    server_receive_time = Column(Integer)

    person_track_id = Column(
        Integer,
        default=-1
    )

    trigger_x = Column(
        Integer,
        default=0
    )

    trigger_y = Column(
        Integer,
        default=0
    )

    keyword = Column(
        String,
        default=""
    )

    # NEW / ACKNOWLEDGED / RESOLVED
    status = Column(
        String,
        default="NEW"
    )

    video_url = Column(
        String,
        default=""
    )

# 设备台账与心跳状态表
class DeviceRecord(Base):
    __tablename__ = "devices"

    device_id = Column(String, primary_key=True, index=True)
    deployment_area = Column(String, default="未配置区域")
    model_version = Column(String, default="YOLOv8n-Pose.rknn")

    cpu_usage = Column(Integer, default=-1)
    memory_usage = Column(Integer, default=-1)
    storage_usage = Column(Integer, default=-1)

    last_online_time = Column(Integer, default=0)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# 2. 全局配置与 MQTT
MQTT_BROKER = "10.48.212.22"
MQTT_PORT = 1883
ALERT_TOPIC = "fall_detection/alerts"
STATUS_TOPIC = "fall_detection/status/#" # 订阅所有设备的心跳主题
CLIENT_ID = "Cloud_Backend_FastAPI_001"
mqtt_client = None

class LiveCommand(BaseModel):
    rtmp_url: str

class AlertStatusUpdate(BaseModel):
    status: str

# 3. MQTT 回调函数 (核心重构)
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] 已连接到云端 Broker，状态码: {rc}")
    client.subscribe(ALERT_TOPIC, qos=1)
    client.subscribe(STATUS_TOPIC, qos=0) # 订阅心跳状态

def on_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    topic = msg.topic
    db = SessionLocal()
    try:
        data = json.loads(payload)
        
        # 【路由 A】：处理设备定时上报的心跳与 NPU 状态
        if topic.startswith("fall_detection/status/"):
            print(f"[MQTT] 收到设备状态心跳: {payload}")

            dev_id = data.get("device_id", "unknown")

            dev = db.query(DeviceRecord).filter(
                DeviceRecord.device_id == dev_id
            ).first()

            if not dev:
                dev = DeviceRecord(device_id=dev_id)
                db.add(dev)

            dev.deployment_area = data.get(
                "deployment_area",
                dev.deployment_area or "未配置区域"
            )

            dev.cpu_usage = data.get("cpu_usage", -1)
            dev.memory_usage = data.get("memory_usage", -1)
            dev.storage_usage = data.get("storage_usage", -1)

            dev.last_online_time = int(time.time() * 1000)

            db.commit()
            
        # 【路由 B】：处理摔倒报警事件
        elif topic == ALERT_TOPIC:
            print(
                f"\n[MQTT] 收到统一报警事件: "
                f"{payload}"
            )

            event_id = data.get("event_id", "")

            if not event_id:
                print("[MQTT] 报警数据缺少 event_id，忽略")
                return

            # event_id 幂等检查
            existing_alert = (
                db.query(AlertRecord)
                .filter(
                    AlertRecord.event_id == event_id
                )
                .first()
            )

            if existing_alert:
                print(
                    f"[MQTT] 重复报警事件，忽略："
                    f"event_id={event_id}"
                )

                return


            event_data = data.get(
                "data",
                {}
            )

            event_id = data.get("event_id", "")
            event_type = data.get("event_type", "UNKNOWN")
            sources = data.get("sources", [])
            event_data = data.get("data", {})
            timestamp = data.get("timestamp", 0)

            fusion_window_ms = 10000

            recent_alert = db.query(AlertRecord).filter(
                AlertRecord.device_id == data.get("device_id", "unknown"),
                AlertRecord.timestamp >= timestamp - fusion_window_ms,
                AlertRecord.timestamp <= timestamp + fusion_window_ms
            ).order_by(AlertRecord.timestamp.desc()).first()

            should_fuse = False

            if recent_alert:
                try:
                    old_sources = json.loads(recent_alert.sources or "[]")
                except Exception:
                    old_sources = []

                incoming_has_vision = "VISION" in sources
                incoming_has_voice = "VOICE" in sources
                old_has_vision = "VISION" in old_sources
                old_has_voice = "VOICE" in old_sources

                should_fuse = (
                    (incoming_has_vision and old_has_voice) or
                    (incoming_has_voice and old_has_vision)
                )

            if should_fuse:
                merged_sources = list(dict.fromkeys(old_sources + sources))

                recent_alert.sources = json.dumps(
                    merged_sources,
                    ensure_ascii=False
                )

                # 只要其中包含视觉跌倒，最终事件按 FALL 记录
                if event_type == "FALL" or recent_alert.event_type == "FALL":
                    recent_alert.event_type = "FALL"

                keyword = event_data.get("keyword", "")
                if keyword:
                    recent_alert.keyword = keyword

                person_track_id = event_data.get("person_track_id", -1)
                if person_track_id >= 0:
                    recent_alert.person_track_id = person_track_id

                trigger_x = event_data.get("trigger_x", 0)
                trigger_y = event_data.get("trigger_y", 0)

                if trigger_x != 0 or trigger_y != 0:
                    recent_alert.trigger_x = trigger_x
                    recent_alert.trigger_y = trigger_y

                deployment_area = data.get("deployment_area", "")
                if deployment_area:
                    recent_alert.deployment_area = deployment_area

                db.commit()

                print(
                    f"[FUSION] 多模态事件融合成功："
                    f"event_id={recent_alert.event_id}, "
                    f"sources={merged_sources}"
                )

                return

            new_alert = AlertRecord(
                event_id=event_id,

                device_id=data.get(
                    "device_id",
                    "unknown"
                ),

                deployment_area=data.get(
                    "deployment_area",
                    "未配置区域"
                ),

                event_type=data.get(
                    "event_type",
                    "UNKNOWN"
                ),

                sources=json.dumps(
                    data.get(
                        "sources",
                        []
                    ),
                    ensure_ascii=False
                ),

                timestamp=data.get(
                    "timestamp",
                    0
                ),

                server_receive_time=int(
                    time.time() * 1000
                ),

                person_track_id=event_data.get(
                    "person_track_id",
                    -1
                ),

                trigger_x=event_data.get(
                    "trigger_x",
                    0
                ),

                trigger_y=event_data.get(
                    "trigger_y",
                    0
                ),

                keyword=event_data.get(
                    "keyword",
                    ""
                ),

                status=data.get(
                    "status",
                    "NEW"
                ),

                video_url=event_data.get(
                    "video_url",
                    ""
                )
            )


            db.add(new_alert)
            db.commit()


            print(
                f"[MQTT] 报警事件已持久化："
                f"event_id={event_id}, "
                f"type={new_alert.event_type}"
            )
            
    except Exception as e:
        print(f"[MQTT] 解析消息失败: {e}")
    finally:
        db.close()

# 4. FastAPI 生命周期与 API
@asynccontextmanager
async def lifespan(app: FastAPI):
    global mqtt_client
    mqtt_client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, CLIENT_ID)
    mqtt_client.on_connect = on_connect
    mqtt_client.on_message = on_message
    try:
        mqtt_client.connect(MQTT_BROKER, MQTT_PORT, 60)
        mqtt_client.loop_start()
    except Exception as e:
        print(f"[MQTT] 连接失败: {e}")
    yield
    if mqtt_client:
        mqtt_client.loop_stop()
        mqtt_client.disconnect()

app = FastAPI(title="摔倒检测云端管理系统", lifespan=lifespan)
VIDEO_DIR = "uploaded_videos"
os.makedirs(VIDEO_DIR, exist_ok=True)

app.mount("/videos", StaticFiles(directory=VIDEO_DIR), name="videos")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/api/alerts")
async def get_alerts(
    limit: int = 50,
    db: Session = Depends(get_db)
):
    alerts = (
        db.query(AlertRecord)
        .order_by(
            AlertRecord.server_receive_time.desc()
        )
        .limit(limit)
        .all()
    )


    res_list = []


    for alert in alerts:
        try:
            sources = json.loads(
                alert.sources or "[]"
            )
        except Exception:
            sources = []


        res_list.append({
            "id":
                alert.id,

            "event_id":
                alert.event_id,

            "device_id":
                alert.device_id,

            "deployment_area":
                alert.deployment_area,

            "event_type":
                alert.event_type,

            "sources":
                sources,

            "timestamp":
                alert.timestamp,

            "server_receive_time":
                alert.server_receive_time,

            "person_track_id":
                alert.person_track_id,

            "trigger_x":
                alert.trigger_x,

            "trigger_y":
                alert.trigger_y,

            "keyword":
                alert.keyword,

            "status":
                alert.status,

            "video_url":
                alert.video_url
        })


    return {
        "code": 200,
        "data": res_list
    }

@app.get("/api/dashboard")
async def get_dashboard(db: Session = Depends(get_db)):
    # 1. 动态判断真实在线设备数（30秒内发过心跳算在线）
    current_time = int(time.time() * 1000)
    online_devices = db.query(DeviceRecord).filter(current_time - DeviceRecord.last_online_time <= 30000).all()
    online_count = len(online_devices)

    current_device = None

    if online_devices:
        current_device = max(
            online_devices,
            key=lambda d: d.last_online_time
        )

    memory_usage = (
        current_device.memory_usage
        if current_device else -1
    )

    storage_usage = (
        current_device.storage_usage
        if current_device else -1
    )
    
    # 3. 统计今日报警数与近7天趋势
    today_start = int(datetime.now().replace(hour=0, minute=0, second=0).timestamp() * 1000)
    today_alerts = db.query(AlertRecord).filter(AlertRecord.server_receive_time >= today_start).count()
    today_fall_alerts = db.query(AlertRecord).filter(
    AlertRecord.server_receive_time >= today_start, AlertRecord.event_type == "FALL").count()

    today_help_alerts = db.query(AlertRecord).filter(
        AlertRecord.server_receive_time >= today_start,
        AlertRecord.event_type == "HELP_REQUEST"
    ).count()
    pending_alerts = (db.query(AlertRecord).filter(AlertRecord.status.in_(["NEW", "ACKNOWLEDGED"])).count())
    
    trend_data = []
    fall_trend_data = []
    help_trend_data = []

    for i in range(6, -1, -1):
        day_start = int((datetime.now() - timedelta(days=i)).replace(hour=0, minute=0, second=0, microsecond=0).timestamp() * 1000)
        day_end = day_start + 86400000

        total_count = db.query(AlertRecord).filter(
            AlertRecord.server_receive_time >= day_start,
            AlertRecord.server_receive_time < day_end
        ).count()

        fall_count = db.query(AlertRecord).filter(
            AlertRecord.server_receive_time >= day_start,
            AlertRecord.server_receive_time < day_end,
            AlertRecord.event_type == "FALL"
        ).count()

        help_count = db.query(AlertRecord).filter(
            AlertRecord.server_receive_time >= day_start,
            AlertRecord.server_receive_time < day_end,
            AlertRecord.event_type == "HELP_REQUEST"
        ).count()

        trend_data.append(total_count)
        fall_trend_data.append(fall_count)
        help_trend_data.append(help_count)

    return {
        "code": 200,
        "data": {
            "online_gateways": online_count,
            "today_alerts": today_alerts,
            "pending_alerts": pending_alerts,
            "memory_usage": memory_usage,
            "storage_usage": storage_usage,
            "trend_7_days": trend_data
        }
    }

@app.get("/api/devices")
async def get_devices(db: Session = Depends(get_db)):
    current_time = int(time.time() * 1000)
    devices = db.query(DeviceRecord).all()
    
    res_list = []
    for d in devices:
        # 判断是否在 30 秒内有过心跳
        is_online = (current_time - d.last_online_time <= 30000)
        res_list.append({
            "device_id": d.device_id,
            "deployment_area": d.deployment_area,
            "status": "🟢在线" if is_online else "🔴离线",
            "model_version": d.model_version,
            "cpu_usage": d.cpu_usage,
            "memory_usage": d.memory_usage,
            "storage_usage": d.storage_usage
        })
        
    return {"code": 200, "data": res_list}

@app.patch("/api/alerts/{event_id}/status")
async def update_alert_status(
    event_id: str,
    request: AlertStatusUpdate,
    db: Session = Depends(get_db)
):
    # 只允许这三种业务状态
    allowed_status = {
        "NEW",
        "ACKNOWLEDGED",
        "RESOLVED"
    }

    if request.status not in allowed_status:
        return {
            "code": 400,
            "message": "非法报警状态"
        }


    alert = (
        db.query(AlertRecord)
        .filter(
            AlertRecord.event_id == event_id
        )
        .first()
    )


    if not alert:
        return {
            "code": 404,
            "message": "报警事件不存在"
        }


    alert.status = request.status

    db.commit()
    db.refresh(alert)


    return {
        "code": 200,
        "message": "报警状态更新成功",
        "data": {
            "event_id": alert.event_id,
            "status": alert.status
        }
    }

@app.post("/api/upload/video")
async def upload_video(request: Request, file: UploadFile = File(...), db: Session = Depends(get_db)):
    filename = os.path.basename(file.filename)
    save_path = os.path.join(VIDEO_DIR, filename)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # 文件名格式：fall_<event_id>.mp4
    event_id = ""

    if filename.startswith("fall_") and filename.endswith(".mp4"):
        event_id = filename[5:-4]

    if not event_id:
        return {
            "code": 400,
            "message": "无法从视频文件名解析 event_id"
        }

    alert = db.query(AlertRecord).filter(AlertRecord.event_id == event_id).first()

    if not alert:
        return {
            "code": 404,
            "message": f"未找到对应报警事件：{event_id}"
        }

    base_url = str(request.base_url).rstrip("/")
    video_url = f"{base_url}/videos/{filename}"

    alert.video_url = video_url
    db.commit()

    print(f"[VIDEO] 视频已关联报警：event_id={event_id}, url={video_url}")

    return {
        "code": 200,
        "message": "视频上传成功",
        "event_id": event_id,
        "video_url": video_url
    }