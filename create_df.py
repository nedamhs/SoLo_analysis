from pathlib import Path
import argparse
import pandas as pd

from feature_extraction import *


DATASET_PATH = Path("/Users/nedamohseni/Downloads/SoLo_dataset")

HOUR_MS = 60 * 60 * 1000

FEATURE_WINDOWS_HOURS = {"calls": 12, 
                         "notifications": 12, 
                         "messages": 12, 
                         "applications_foreground": 12,
                         "battery_charges": 12,
                         "touch": 12, 
                         "screen": 12,
                         "device_usage": 12, 
                         "studentlife_audio": 12, 
                         "ambient_noise": 12, 
                         "samsung": 24}

FEATURE_WINDOWS = {name: hours * HOUR_MS for name, hours in FEATURE_WINDOWS_HOURS.items()}


def load_ema(participant_dir):
    ema_path = participant_dir / "Self_Report" / "ema_daily.csv"
    ema = pd.read_csv(ema_path)
    metadata_cols = ["participant", "date", "timestamp", "surveyindex"]
    ema = ema.rename(columns={col: f"EMA_{col}" for col in ema.columns if col not in metadata_cols})
    return ema

def load_aware(participant_dir):
    aware_dir = participant_dir / "AWARE"

    calls = pd.read_csv(aware_dir / "calls.csv") if (aware_dir / "calls.csv").exists() else None
    messages = pd.read_csv(aware_dir / "messages.csv") if (aware_dir / "messages.csv").exists() else None
    notifications = pd.read_csv(aware_dir / "notifications.csv") if (aware_dir / "notifications.csv").exists() else None
    applications_foreground = pd.read_csv(aware_dir / "applications_foreground.csv") if (aware_dir / "applications_foreground.csv").exists() else None
    screen = pd.read_csv(aware_dir / "screen.csv") if (aware_dir / "screen.csv").exists() else None
    touch = pd.read_csv(aware_dir / "touch.csv") if (aware_dir / "touch.csv").exists() else None
    battery_charges = pd.read_csv(aware_dir / "battery_charges.csv") if (aware_dir / "battery_charges.csv").exists() else None
    device_usage = pd.read_csv(aware_dir / "device_usage.csv") if (aware_dir / "device_usage.csv").exists() else None
    studentlife_audio = pd.read_csv(aware_dir / "studentlife_audio.csv") if (aware_dir / "studentlife_audio.csv").exists() else None
    ambient_noise = pd.read_csv(aware_dir / "ambient_noise.csv") if (aware_dir / "ambient_noise.csv").exists() else None
    mobility = pd.read_csv(aware_dir / "mobility_features.csv") if (aware_dir / "mobility_features.csv").exists() else None

    return calls, messages, notifications, applications_foreground, screen, touch, battery_charges, device_usage, studentlife_audio, ambient_noise, mobility
    
def load_hrv(participant_dir):
    hrv_path = participant_dir / "Samsung" / "hrv_5min.csv"
    hrv = pd.read_csv(hrv_path) if hrv_path.exists() else None
    return hrv

def load_oura(participant_dir):
    oura_dir = participant_dir / "Oura"

    activity_daily = pd.read_csv(oura_dir / "activity_daily.csv") if (oura_dir / "activity_daily.csv").exists() else None
    sleep_daily = pd.read_csv(oura_dir / "sleep_daily.csv") if (oura_dir / "sleep_daily.csv").exists() else None
    readiness_daily = pd.read_csv(oura_dir / "readiness_daily.csv") if (oura_dir / "readiness_daily.csv").exists() else None

    return activity_daily, sleep_daily, readiness_daily


def create_participant_df(dataset_path, participant):
    participant_dir = dataset_path / participant

    # load EMA
    df = load_ema(participant_dir)
    # load AWARE
    calls, messages, notifications, applications_foreground, screen, touch, battery_charges, device_usage, studentlife_audio, ambient_noise, mobility = load_aware(participant_dir)
    # load hrv features
    hrv = load_hrv(participant_dir)
    #load oura 
    activity_daily, sleep_daily, readiness_daily = load_oura(participant_dir)

    rows = []

    for _, ema_row in df.iterrows():
        call_features = extract_calls_features(calls, ema_row, FEATURE_WINDOWS)
        message_features = extract_messages_features(messages, ema_row, FEATURE_WINDOWS)
        notification_features = extract_notifications_features(notifications, ema_row, FEATURE_WINDOWS)
        foreground_features = extract_applications_foreground_features(applications_foreground, ema_row, FEATURE_WINDOWS)
        screen_features = extract_screen_features(screen, ema_row, FEATURE_WINDOWS)
        touch_features = extract_touch_features(touch, ema_row, FEATURE_WINDOWS)
        battery_features = extract_battery_charges_features(battery_charges, ema_row, FEATURE_WINDOWS)
        device_usage_features = extract_device_usage_features(device_usage, ema_row, FEATURE_WINDOWS)
        studentlife_audio_features = extract_studentlife_audio_features(studentlife_audio, ema_row, FEATURE_WINDOWS)
        ambient_noise_features = extract_ambient_noise_features(ambient_noise, ema_row, FEATURE_WINDOWS)
        mobility_features = get_mobility_features(mobility, ema_row)

        hrv_features = aggregate_hrv_features(hrv, ema_row, FEATURE_WINDOWS)
        oura_features = get_oura_features(activity_daily, sleep_daily, readiness_daily, ema_row)
        

        row = ema_row.to_dict()
        row.update({f"AWARE_{name}": value for name, value in call_features.items()})
        row.update({f"AWARE_{name}": value for name, value in message_features.items()})
        row.update({f"AWARE_{name}": value for name, value in notification_features.items()})
        row.update({f"AWARE_{name}": value for name, value in foreground_features.items()})
        row.update({f"AWARE_{name}": value for name, value in screen_features.items()})
        row.update({f"AWARE_{name}": value for name, value in touch_features.items()})
        row.update({f"AWARE_{name}": value for name, value in battery_features.items()})
        row.update({f"AWARE_{name}": value for name, value in device_usage_features.items()})
        row.update({f"AWARE_{name}": value for name, value in studentlife_audio_features.items()})
        row.update({f"AWARE_{name}": value for name, value in ambient_noise_features.items()})
        row.update({f"AWARE_{name}": value for name, value in mobility_features.items()})
        row.update({f"SAMSUNG_{name}": value for name, value in hrv_features.items()})
        row.update({f"OURA_{name}": value for name, value in oura_features.items()})
        rows.append(row)

    return pd.DataFrame(rows)


def create_all_participants_df(dataset_path):
    participants = sorted([path.name for path in dataset_path.iterdir() if path.is_dir() and path.name.startswith("pers")])
    participant_dfs = []

    for participant in participants:
        print(f"Processing {participant}...")
        participant_df = create_participant_df(dataset_path, participant)
        participant_dfs.append(participant_df)

    return pd.concat(participant_dfs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("solo_dataframe.pkl"))
    args = parser.parse_args()
    df = create_all_participants_df(args.dataset_path)
    df.to_pickle(args.output)
    print(f"Saved {len(df)} rows and {len(df.columns)} columns to {args.output}")


if __name__ == "__main__":
    main()


