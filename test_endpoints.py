import subprocess
import time
import requests
import sys

def run_tests():
    # Start the Flask app
    print("Starting Flask app...")
    server_process = subprocess.Popen(
        [sys.executable, "app.py"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    time.sleep(3)
    
    base_url = "http://127.0.0.1:5000"
    
    tests = [
        {
            "name": "Crop Prediction - Valid",
            "url": f"{base_url}/api/predict-crop",
            "payload": {
                "N": 50,
                "P": 50,
                "K": 50,
                "temperature": 25.0,
                "humidity": 60.0,
                "ph": 6.5,
                "rainfall": 100.0
            },
            "expect_success": True
        },
        {
            "name": "Crop Prediction - Missing Field",
            "url": f"{base_url}/api/predict-crop",
            "payload": {
                "N": 50,
                "P": 50
            },
            "expect_success": False
        },
        {
            "name": "Fertilizer Prediction - Valid",
            "url": f"{base_url}/api/predict-fertilizer",
            "payload": {
                "temperature": 26,
                "humidity": 52,
                "moisture": 38,
                "soil_type": "Sandy",
                "crop_type": "Maize",
                "nitrogen": 37,
                "potassium": 0,
                "phosphorous": 0
            },
            "expect_success": True
        },
        {
            "name": "Fertilizer Prediction - Invalid Soil Type",
            "url": f"{base_url}/api/predict-fertilizer",
            "payload": {
                "temperature": 26,
                "humidity": 52,
                "moisture": 38,
                "soil_type": "SuperSandy", # invalid category
                "crop_type": "Maize",
                "nitrogen": 37,
                "potassium": 0,
                "phosphorous": 0
            },
            "expect_success": False
        },
        {
            "name": "Crop Type Prediction - Valid",
            "url": f"{base_url}/api/predict-type",
            "payload": {
                "temperature": 26,
                "humidity": 52,
                "moisture": 38,
                "soil_type": "Sandy",
                "nitrogen": 37,
                "potassium": 0,
                "phosphorous": 0
            },
            "expect_success": True
        }
    ]
    
    all_passed = True
    print("\n=== RUNNING INTEGRATION TESTS ===")
    
    for test in tests:
        print(f"\nRunning test: {test['name']}")
        try:
            response = requests.post(test['url'], json=test['payload'], timeout=5)
            if response.status_code != 200:
                print(f"[FAIL] Expected status code 200, got {response.status_code}")
                all_passed = False
                continue
                
            res_json = response.json()
            if res_json.get("success") == test["expect_success"]:
                print(f"[PASS] Response matched expectation. Success={res_json.get('success')}")
                if res_json.get("success"):
                    print(f"       Prediction: {res_json.get('prediction')} (Confidence: {res_json.get('confidence')}%)")
                else:
                    print(f"       Error Message: {res_json.get('error')}")
            else:
                print(f"[FAIL] Expected success={test['expect_success']}, got success={res_json.get('success')}")
                print(f"       Response content: {res_json}")
                all_passed = False
        except Exception as e:
            print(f"[FAIL] Request failed: {e}")
            all_passed = False
            
    # Terminate the server
    print("\nStopping Flask app...")
    server_process.terminate()
    try:
        server_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        server_process.kill()
        
    print("\n=================================")
    if all_passed:
        print("ALL TESTS PASSED SUCCESSFULLY!")
        sys.exit(0)
    else:
        print("SOME TESTS FAILED.")
        sys.exit(1)

if __name__ == "__main__":
    run_tests()
