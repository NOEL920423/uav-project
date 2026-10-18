#!/bin/bash
echo "nas backup practice"
echo "this is my first bash script"
SOURCE_ROOT="/home/noel_614420090/uav-project"
NAS_MOUNT="/home/noel_614420090/NAS"
BACKUP_DIR="$NAS_MOUNT/614/614420090_ChenHao/uav-project"
echo "來源：$SOURCE_ROOT"
echo "掛載點：$NAS_MOUNT"
echo "位置：$BACKUP_DIR"

# 檢查掛載點
if mountpoint -q "$NAS_MOUNT"; then
    echo "NAS is mounted."
    else
    echo "NAS isn't mounted."
    exit 1
    fi 

# 檢查資料存在嗎
for DIR_NAME in datasets experiments evaluations; do 
    if [[ -d "$SOURCE_ROOT/$DIR_NAME" ]]; then
        echo "$DIR_NAME 存在"
        else
        echo "$DIR_NAME不存在"
        exit 1
        fi
done

# 執行 dry-run
if rsync -rt --omit-dir-times --dry-run --stats "$SOURCE_ROOT/experiments" \
"$SOURCE_ROOT/datasets" \
"$SOURCE_ROOT/evaluations" \
"$BACKUP_DIR/"; then
    echo "dry-run completed"
    else
    echo "dry-run failed"
    exit 1
    fi

if read -r -p "確定要備份到 NAS 嗎，yes or no： " ANSWER; then
    echo "你輸入： $ANSWER"
    if [[ "$ANSWER" == "yes" ]]; then
        echo "讀取成功，開始備份"
        if rsync -rt --omit-dir-times "$SOURCE_ROOT/experiments" \
        "$SOURCE_ROOT/datasets" \
        "$SOURCE_ROOT/evaluations" \
        "$BACKUP_DIR/"; then
            echo "備份完成"
            else
            echo "備份失敗"
            exit 1
            fi
        else
        echo "備份取消"
        exit 0
        fi
    else
    echo "讀取失敗"
    exit 1
    fi