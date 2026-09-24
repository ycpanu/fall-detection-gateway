import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from datetime import datetime, timedelta

# ==========================================
# 1. 数据库配置 (SQLite)
# ==========================================
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
        nullable=False
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
    location = Column(String, default="未分配位置")
    model_version = Column(String, default="YOLOv8n-Pose.rknn")
    npu_usage = Column(Integer, default=0)
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
            print(f"[MQTT] 收到设备心跳与状态: {payload}")
            dev_id = data.get("device_id", "unknown")
            dev = db.query(DeviceRecord).filter(DeviceRecord.device_id == dev_id).first()
            if not dev:
                dev = DeviceRecord(device_id=dev_id, location="客厅测试点")
                db.add(dev)
            
            dev.npu_usage = data.get("npu_usage", 0)
            dev.last_online_time = int(time.time() * 1000) # 更新最后存活时间
            db.commit()
            
        # 【路由 B】：处理摔倒报警事件
        elif topic == ALERT_TOPIC:
            print(
                f"\n[MQTT] 收到统一报警事件: "
                f"{payload}"
            )

            event_id = data.get(
                "event_id",
                ""
            )

            if not event_id:
                print(
                    "[MQTT] 报警数据缺少 event_id，忽略"
                )
                return


            # =====================================
            # event_id 幂等检查
            # =====================================

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


            new_alert = AlertRecord(
                event_id=event_id,

                device_id=data.get(
                    "device_id",
                    "unknown"
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
    
    # 2. 动态计算真实 NPU 平均利用率
    avg_npu = int(sum(d.npu_usage for d in online_devices) / online_count) if online_count > 0 else 0
    
    # 3. 统计今日报警数与近7天趋势
    today_start = int(datetime.now().replace(hour=0, minute=0, second=0).timestamp() * 1000)
    today_alerts = db.query(AlertRecord).filter(AlertRecord.server_receive_time >= today_start).count()
    
    trend_data = []
    for i in range(6, -1, -1):
        day_start = int((datetime.now() - timedelta(days=i)).replace(hour=0, minute=0, second=0).timestamp() * 1000)
        day_end = day_start + 86400000
        count = db.query(AlertRecord).filter(
            AlertRecord.server_receive_time >= day_start,
            AlertRecord.server_receive_time < day_end
        ).count()
        trend_data.append(count)

    return {
        "code": 200,
        "data": {
            "online_gateways": online_count,       # 真数据！
            "today_alerts": today_alerts,          # 真数据！
            "pending_alerts": 0,
            "npu_usage": f"{avg_npu} %",           # 真数据！
            "trend_7_days": trend_data             # 真数据！
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
            "location": d.location,
            "status": "🟢 在线" if is_online else "🔴 离线",
            "model_version": d.model_version
        })
        
    return {"code": 200, "data": res_list}