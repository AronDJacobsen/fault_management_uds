In the **SCADA iFIX** system, data quality stamping refers to a mechanism that assigns a "quality" indicator to the data being acquired and monitored from field devices and sensors. This quality stamp reflects the reliability, validity, and overall trustworthiness of the data, which is crucial for decision-making in industrial automation environments.

The quality stamping in iFIX SCADA is typically based on the following factors:

### 1. **Communication Status**
   - The most common determinant of data quality is the communication status between the SCADA system and the field device (like PLCs, RTUs, or sensors). If communication is interrupted, lost, or unstable, iFIX assigns a "bad" or "questionable" quality stamp to the data.

### 2. **Device or Sensor Health**
   - The health of the device or sensor transmitting the data plays a role in the quality stamp. If the sensor is malfunctioning, experiencing errors, or sending out-of-range values, the data quality will be affected accordingly.

### 3. **Data Freshness**
   - This refers to how recent the data is. If a piece of data has not been updated within a specified time frame, iFIX will stamp it as "stale," meaning it is potentially outdated and might not represent the current state of the system.

### 4. **Out of Range or Invalid Values**
   - Some sensors or devices may send data that is out of the expected range or represent values that are physically or logically impossible. iFIX uses configured limits and checks to ensure the data is within valid ranges, and if not, the data is flagged as having "bad" quality.

### 5. **Manual Override**
   - If data is manually overridden by an operator for specific reasons (e.g., calibration, maintenance), the system can mark this data with a quality stamp indicating the manual intervention, distinguishing it from automatic real-time data.

### 6. **Driver Quality**
   - iFIX interfaces with a wide variety of drivers (OPC, native protocols) that connect it to field devices. If there are issues with the driver (e.g., software malfunction or protocol errors), this can impact data quality, and the iFIX system will mark the data as "bad" or "questionable."

### 7. **Redundancy Status**
   - For SCADA systems configured with redundancy (e.g., dual servers or backup systems), if the primary system fails and data is being retrieved from a backup system, the data quality can be stamped based on the redundancy status.

### Common Quality Stamps in iFIX
   iFIX typically uses the following quality categories:
   - **Good**: Data is valid, reliable, and timely.
   - **Bad**: Data is not reliable (due to communication issues, device failure, etc.).
   - **Uncertain/Questionable**: Data may be partially valid but not fully trustworthy, often due to temporary issues in communication or device operation.
   - **Stale**: Data has not been updated for a while, and therefore might not reflect the current state.

### OPC Quality Model
   iFIX often interfaces with OPC servers to retrieve data from field devices. The OPC standard has its own quality model, and iFIX maps its internal data quality stamps to this OPC quality structure. The typical OPC quality codes are:
   - **Good**: Data is fully valid.
   - **Uncertain**: Data may be valid but has uncertainties, such as sensor drift or intermittent communication.
   - **Bad**: Data is invalid due to device faults or communication loss.

### Summary:
iFIX assigns quality stamps based on various criteria including communication status, sensor/device health, data freshness, data validity (in-range or out-of-range values), and any manual overrides. The goal is to provide operators and systems with clear indicators of data trustworthiness to ensure safe and efficient operation of industrial processes.

Would you like more detailed information on how to configure or interpret these quality stamps in iFIX?