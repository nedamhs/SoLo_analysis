import pandas as pd

APPLICATION_CATEGORIES = ["Shopping", "Entertainment", "Tools", "Travel & Local", "Lifestyle", "Social", "Auto & Vehicles", "Education", "Business", "Finance", "Health & Fitness", "Music & Audio", "Productivity", "Books & Reference", "Photography", "Video Players & Editors", "Communication", "Unknown"]


def extract_calls_features(calls, datapoint, feature_windows):
    """for each call type (1-incoming/2-outgoing/3-missed), 
            - sum of duration in the time window for incoming(1) and outgoing calls(2) only
            - num of calls in the time window for each type (incoming, ougoing, missed)"""
            
    if calls is None or calls.empty:
        return {"call_incoming_duration": None, "call_outgoing_duration": None, "call_incoming_count": None, "call_outgoing_count": None, "call_missed_count": None}

    part_calls = calls[(calls["timestamp"] < datapoint["timestamp"]) & (calls["timestamp"] > datapoint["timestamp"] - feature_windows["calls"])]
    call_durs = part_calls.groupby("call_type")["call_duration"].sum().to_dict()
    call_counts = part_calls.groupby("call_type").size().to_dict()
    features_calls = {"call_incoming_duration": call_durs.get(1, 0), "call_outgoing_duration": call_durs.get(2, 0), "call_incoming_count": call_counts.get(1, 0), "call_outgoing_count": call_counts.get(2, 0), "call_missed_count": call_counts.get(3, 0)}

    return features_calls

def extract_messages_features(messages, datapoint, feature_windows):
    """Count messages per type(1-received, 2-sent) within the time window."""

    if messages is None or messages.empty:
        return {"message_received_count": None, "message_sent_count": None}

    part_messages = messages[(messages["timestamp"] < datapoint["timestamp"]) & (messages["timestamp"] > datapoint["timestamp"] - feature_windows["messages"])]
    message_count = part_messages.groupby("message_type").size().to_dict()
    features_messages = {"message_received_count": message_count.get(1, 0), "message_sent_count": message_count.get(2, 0)}

    return features_messages

def extract_notifications_features(notifications, datapoint, feature_windows):
    """Count notifications per category in the time window
    (Shopping, Entertainment, Tools, Travel & Local, Lifestyle, Social, Auto & Vehicles, Education, Business, Finance,
    Health & Fitness, Music & Audio, Productivity, Books & Reference, Photography, Video Players & Editors, Communication, Unknown)."""

    if notifications is None or notifications.empty:
        return {f"notification_{category}": None for category in APPLICATION_CATEGORIES}

    part_notifications = notifications[(notifications["timestamp"] < datapoint["timestamp"]) & (notifications["timestamp"] > datapoint["timestamp"] - feature_windows["notifications"])]
    notif_count = part_notifications.groupby("application_category").size().to_dict()
    features_notifications = {f"notification_{category}": notif_count.get(category, 0) for category in APPLICATION_CATEGORIES}

    return features_notifications

def extract_applications_foreground_features(applications_foreground, datapoint, feature_windows):
    """Count non-system foreground application records per category in the time window
    (Shopping, Entertainment, Tools, Travel & Local, Lifestyle, Social, Auto & Vehicles, Education, Business, Finance,
    Health & Fitness, Music & Audio, Productivity, Books & Reference, Photography, Video Players & Editors, Communication, Unknown)."""

    if applications_foreground is None or applications_foreground.empty:
        return {f"foreground_{category}": None for category in APPLICATION_CATEGORIES}

    non_system_foreground = applications_foreground[applications_foreground["is_system_app"] == 0]
    part_foreground = non_system_foreground[(non_system_foreground["timestamp"] < datapoint["timestamp"]) & (non_system_foreground["timestamp"] > datapoint["timestamp"] - feature_windows["applications_foreground"])]
    non_system_foreground_count = part_foreground.groupby("application_category").size().to_dict()
    features_foreground = {f"foreground_{category}": non_system_foreground_count.get(category, 0) for category in APPLICATION_CATEGORIES}
        
    return features_foreground

def extract_screen_features(screen, datapoint, feature_windows):
    """Count screen events per status within the time window (0=off, 1=on, 2=locked, 3=unlocked)."""

    if screen is None or screen.empty:
        return {"screen_off_count": None, "screen_on_count": None, "screen_locked_count": None, "screen_unlocked_count": None}

    part_screen = screen[(screen["timestamp"] < datapoint["timestamp"]) & (screen["timestamp"] > datapoint["timestamp"] - feature_windows["screen"])]
    screen_counts = part_screen.groupby("screen_status").size().to_dict()
    features_screen = {"screen_off_count": screen_counts.get(0, 0), "screen_on_count": screen_counts.get(1, 0), "screen_locked_count": screen_counts.get(2, 0), "screen_unlocked_count": screen_counts.get(3, 0)}

    return features_screen

def extract_touch_features(touch, datapoint, feature_windows):
    """Count touch events per action type within the time window (0=clicked, 1=long_clicked, 2=scrolled_up, 3=scrolled_down)."""

    if touch is None or touch.empty:
        return {"touch_clicked_count": None, "touch_long_clicked_count": None, "touch_scrolled_up_count": None, "touch_scrolled_down_count": None}

    part_touch = touch[(touch["timestamp"] < datapoint["timestamp"]) & (touch["timestamp"] > datapoint["timestamp"] - feature_windows["touch"])]
    touch_counts = part_touch.groupby("touch_action").size().to_dict()
    features_touch = {"touch_clicked_count": touch_counts.get(0, 0), "touch_long_clicked_count": touch_counts.get(1, 0), "touch_scrolled_up_count": touch_counts.get(2, 0), "touch_scrolled_down_count": touch_counts.get(3, 0)}

    return features_touch

def extract_battery_charges_features(battery_charges, datapoint, feature_windows):
    """Return mean battery levels at charging start and end, and number of charging events within the time window."""

    if battery_charges is None or battery_charges.empty:
        return {"battery_charge_start_mean": None, "battery_charge_end_mean": None, "battery_charge_count": None}

    part_battery_charges = battery_charges[(battery_charges["timestamp"] < datapoint["timestamp"]) & (battery_charges["timestamp"] > datapoint["timestamp"] - feature_windows["battery_charges"])]
    mean_start = part_battery_charges["battery_charge_start"].mean()
    mean_end = part_battery_charges["battery_charge_end"].mean()
    count_start = part_battery_charges["battery_charge_start"].count()
    features_battery_charges = {"battery_charge_start_mean": mean_start, "battery_charge_end_mean": mean_end, "battery_charge_count": count_start}

    return features_battery_charges

def extract_device_usage_features(device_usage, datapoint, feature_windows):
    """Return total elapsed device ON duration, total elapsed device OFF duration, number of ON intervals, and number of OFF intervals within the time window."""

    if device_usage is None or device_usage.empty:
        return {"elapsed_device_on_total": None, "elapsed_device_off_total": None, "device_on_count": None, "device_off_count": None}

    part_device_usage = device_usage[(device_usage["timestamp"] < datapoint["timestamp"]) & (device_usage["timestamp"] > datapoint["timestamp"] - feature_windows["device_usage"])]
    total_on = part_device_usage["elapsed_device_on"].sum()
    total_off = part_device_usage["elapsed_device_off"].sum()
    count_on = (part_device_usage["elapsed_device_on"] > 0).sum()
    count_off = (part_device_usage["elapsed_device_off"] > 0).sum()
    features_device_usage = {"elapsed_device_on_total": total_on, "elapsed_device_off_total": total_off, "device_on_count": int(count_on), "device_off_count": int(count_off)}

    return features_device_usage

def extract_studentlife_audio_features(studentlife_audio, datapoint, feature_windows):
    """Return mean acoustic energy, number of voice records, number of detected conversations, and total conversation duration within the time window."""

    if studentlife_audio is None or studentlife_audio.empty:
        return {"audio_energy_mean": None, "audio_voice_count": None, "audio_conversation_count": None, "audio_conversation_duration_total": None}

    part_audio = studentlife_audio[(studentlife_audio["timestamp"] < datapoint["timestamp"]) & (studentlife_audio["timestamp"] > datapoint["timestamp"] - feature_windows["studentlife_audio"])]

    if part_audio.empty:
        return {"audio_energy_mean": None, "audio_voice_count": 0, "audio_conversation_count": 0, "audio_conversation_duration_total": 0}

    energy_mean = part_audio["energy"].mean()
    voice_count = (part_audio["inference"] == 2).sum()

    conversations = part_audio[(part_audio["datatype"] == 2) & part_audio["convo_start"].notna() & part_audio["convo_end"].notna()]
    conversation_count = len(conversations)
    conversation_durations = conversations["convo_end"] - conversations["convo_start"]
    conversation_durations = conversation_durations[conversation_durations >= 0]
    conversation_duration_total = conversation_durations.sum()

    features_studentlife_audio = {"audio_energy_mean": energy_mean, "audio_voice_count": int(voice_count), "audio_conversation_count": int(conversation_count), "audio_conversation_duration_total": conversation_duration_total}

    return features_studentlife_audio

def extract_ambient_noise_features(ambient_noise, datapoint, feature_windows):
    """Return mean sound frequency, mean sound level (dB), and mean RMS of sound within the time window."""

    if ambient_noise is None or ambient_noise.empty:
        return {"ambient_frequency_mean": None, "ambient_decibels_mean": None, "ambient_rms_mean": None}

    part_ambient_noise = ambient_noise[(ambient_noise["timestamp"] < datapoint["timestamp"]) & (ambient_noise["timestamp"] > datapoint["timestamp"] - feature_windows["ambient_noise"])]
    frequency_mean = part_ambient_noise["frequency"].mean()
    decibels_mean = part_ambient_noise["decibels"].mean()
    rms_mean = part_ambient_noise["rms"].mean()
    features_ambient_noise = {"ambient_frequency_mean": frequency_mean, "ambient_decibels_mean": decibels_mean, "ambient_rms_mean": rms_mean}

    return features_ambient_noise

def get_mobility_features(mobility, datapoint):
    """Return mobility features from the calendar day preceding the EMA, excluding the home label and data coverage variables."""

    if mobility is None or mobility.empty:
        return {}

    feature_cols = [col for col in mobility.columns if col not in ["participant", "date", "homelabel", "minutesdataused"]]
    previous_date = (pd.to_datetime(datapoint["date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    part_mobility = mobility[mobility["date"] == previous_date]

    if part_mobility.empty:
        return {f"mobility_{col}": None for col in feature_cols}

    features_mobility = {f"mobility_{col}": part_mobility.iloc[0][col] for col in feature_cols}

    return features_mobility

def aggregate_hrv_features(hrv, datapoint, feature_windows):
    """Return mean HRV features across 5-minute segments within the time window."""

    if hrv is None or hrv.empty:
        return {}

    feature_cols = [col for col in hrv.columns if col not in ["participant", "timestamp"]]
    part_hrv = hrv[(hrv["timestamp"] < datapoint["timestamp"]) & (hrv["timestamp"] > datapoint["timestamp"] - feature_windows["samsung"])]

    if part_hrv.empty:
        return {col: None for col in feature_cols}

    features_hrv = part_hrv[feature_cols].mean().to_dict()

    return features_hrv

def get_oura_features(activity_daily, sleep_daily, readiness_daily, datapoint):
    """Return Oura activity, sleep, and readiness features from the calendar day preceding the EMA."""

    features_oura = {}
    previous_date = (pd.to_datetime(datapoint["date"]) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    exclude_cols = ["participant", "timestamp", "date", "activity_day_start", "activity_day_end", "sleep_bedtime_start", "sleep_bedtime_end", "ideal_bedtime_bedtime_window_start", "ideal_bedtime_bedtime_window_end"]

    for table in [activity_daily, sleep_daily, readiness_daily]:
        if table is None or table.empty:
            continue

        feature_cols = [col for col in table.columns if col not in exclude_cols]
        part_table = table[table["date"] == previous_date]

        if part_table.empty:
            features_oura.update({col: None for col in feature_cols})
        else:
            features_oura.update({col: part_table.iloc[0][col] for col in feature_cols})

    return features_oura