import json
import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import paho.mqtt.client as mqtt

# 引入 SQLAlchemy 相关库
from sqlalchemy import create_engine, Column, Integer, String
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==========================================
# 1. 数据库配置 (SQLite)
# ==========================================
# 数据库文件会生成在当前目录下的 cloud_alerts.db
SQLALCHEMY_DATABASE_URL = "sqlite:///./cloud_alerts.db"

# check_same_thread=False 是 SQLite 在 FastAPI 多线程下必须的参数
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# 定义数据库表结构
class AlertRecord(Base):
    __tablename__ = "alerts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    device_id = Column(String, index=True)
    timestamp = Column(Integer)
    server_receive_time = Column(Integer)
    trigger_x = Column(Integer)
    trigger_y = Column(Integer)
    status = Column(String, default="CRITICAL")

# 自动在本地创建表（如果表不存在）
Base.metadata.create_all(bind=engine)

# 获取数据库 Session 的依赖函数
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==========================================
# 2. 全局配置与 MQTT
# ==========================================
MQTT_BROKER = "broker.emqx.io"
MQTT_PORT = 1883
ALERT_TOPIC = "fall_detection/alerts"
CLIENT_ID = "Cloud_Backend_FastAPI_001"
mqtt_client = None

# ==========================================
# 3. Pydantic 数据模型 (用于接口校验)
# ==========================================
class LiveCommand(BaseModel):
    rtmp_url: str

# ==========================================
# 4. MQTT 回调函数
# ==========================================
def on_connect(client, userdata, flags, rc):
    print(f"[MQTT] 已连接到云端 Broker，状态码: {rc}")
    client.subscribe(ALERT_TOPIC, qos=1)

def on_message(client, userdata, msg):
    payload = msg.payload.decode('utf-8')
    print(f"\n[MQTT] 收到设备报警数据: {payload}")
    try:
        data = json.loads(payload)
        
        # 收到 MQTT 消息后，打开一个独立的数据库会话存入数据
        db = SessionLocal()
        new_alert = AlertRecord(
            device_id=data.get("device_id", "unknown"),
            timestamp=data.get("timestamp", 0),
            server_receive_time=int(time.time() * 1000),
            trigger_x=data.get("data", {}).get("trigger_x", 0),
            trigger_y=data.get("data", {}).get("trigger_y", 0),
            status=data.get("data", {}).get("status", "CRITICAL")
        )
        db.add(new_alert)
        db.commit()
        db.close()
        print("[数据库] 报警数据已成功落盘！")
        
    except Exception as e:
        print(f"[MQTT] 解析或存储报警数据失败: {e}")

# ==========================================
# 5. FastAPI 生命周期
# ==========================================
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

# ==========================================
# 6. RESTful API 路由接口
# ==========================================
@app.get("/")
async def root():
    return {"message": "摔倒检测云端服务运行正常", "status": "ok"}

@app.get("/api/alerts", summary="获取历史报警记录")
async def get_alerts(limit: int = 50, db: Session = Depends(get_db)):
    """从 SQLite 数据库中读取最新的报警记录"""
    # 按接收时间倒序排列，取最新的 limit 条
    alerts = db.query(AlertRecord).order_by(AlertRecord.server_receive_time.desc()).limit(limit).all()
    return {
        "code": 200,
        "msg": "success",
        "total": len(alerts),
        "data": alerts
    }

@app.post("/api/device/{device_id}/live/start")
async def start_device_live(device_id: str, cmd_data: LiveCommand):
    if not mqtt_client: raise HTTPException(status_code=500, detail="MQTT 未初始化")
    topic = f"fall_detection/commands/{device_id}"
    mqtt_client.publish(topic, json.dumps({"cmd": "start_live", "rtmp_url": cmd_data.rtmp_url}), qos=1)
    return {"code": 200, "msg": f"已向设备 {device_id} 下发启动推流指令"}

@app.post("/api/device/{device_id}/live/stop")
async def stop_device_live(device_id: str):
    if not mqtt_client: raise HTTPException(status_code=500, detail="MQTT 未初始化")
    topic = f"fall_detection/commands/{device_id}"
    mqtt_client.publish(topic, json.dumps({"cmd": "stop_live"}), qos=1)
    return {"code": 200, "msg": f"已向设备 {device_id} 下发停止推流指令"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)