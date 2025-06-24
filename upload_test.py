#!/usr/bin/env python3
"""
Simple upload test to diagnose file upload issues
"""

import requests
import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import joblib

def create_test_files():
    """Create simple test files for upload testing"""
    print("Creating test files...")
    
    # Create simple test dataset
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'target': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv('upload_test_data.csv', index=False)
    print(f"Created test dataset: upload_test_data.csv ({df.shape})")
    
    # Create simple test model
    X = pd.get_dummies(df[['age', 'income', 'gender']])
    y = df['target']
    
    model = RandomForestClassifier(n_estimators=5, random_state=42)
    model.fit(X, y)
    
    joblib.dump(model, 'upload_test_model.pkl')
    print("Created test model: upload_test_model.pkl")
    
    return 'upload_test_data.csv', 'upload_test_model.pkl'

def test_server_connection():
    """Test basic server connectivity"""
    print("\n=== Testing Server Connection ===")
    
    try:
        response = requests.get('http://127.0.0.1:5000', timeout=5)
        print(f"✅ Server accessible: {response.status_code}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"❌ Server not accessible: {str(e)}")
        return False

def test_get_columns_endpoint(dataset_file):
    """Test the /get_columns endpoint specifically"""
    print("\n=== Testing /get_columns Endpoint ===")
    
    try:
        with open(dataset_file, 'rb') as f:
            files = {'dataset_file': f}
            
            print("Sending request to /get_columns...")
            response = requests.post(
                'http://127.0.0.1:5000/get_columns',
                files=files,
                timeout=30
            )
            
            print(f"Response status: {response.status_code}")
            print(f"Response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                result = response.json()
                print(f"Response data: {result}")
                
                if result.get('success'):
                    print(f"✅ Columns retrieved: {result.get('columns')}")
                    return True
                else:
                    print(f"❌ Error in response: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response text: {response.text}")
                return False
                
    except Exception as e:
        print(f"❌ Request failed: {str(e)}")
        return False

def test_full_upload(dataset_file, model_file):
    """Test the full upload and analysis process"""
    print("\n=== Testing Full Upload Process ===")
    
    try:
        with open(dataset_file, 'rb') as df, open(model_file, 'rb') as mf:
            files = {
                'dataset_file': df,
                'model_file': mf
            }
            
            data = {
                'target_column': 'target',
                'sensitive_features': ['gender']
            }
            
            print("Sending request to /analyze...")
            response = requests.post(
                'http://127.0.0.1:5000/analyze',
                files=files,
                data=data,
                timeout=60
            )
            
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print("✅ Full upload and analysis successful")
                    return True
                else:
                    print(f"❌ Analysis failed: {result.get('error')}")
                    return False
            else:
                print(f"❌ HTTP Error: {response.status_code}")
                print(f"Response text: {response.text[:500]}")
                return False
                
    except Exception as e:
        print(f"❌ Upload failed: {str(e)}")
        return False

def test_browser_simulation():
    """Test what a browser would send"""
    print("\n=== Testing Browser-like Request ===")
    
    try:
        # Test with browser-like headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-origin'
        }
        
        # Test simple GET first
        response = requests.get('http://127.0.0.1:5000', headers=headers, timeout=5)
        print(f"GET request status: {response.status_code}")
        
        # Test if JavaScript is being served correctly
        js_response = requests.get('http://127.0.0.1:5000/static/script.js', headers=headers, timeout=5)
        print(f"JavaScript file status: {js_response.status_code}")
        
        if js_response.status_code == 200:
            print("✅ JavaScript file accessible")
        else:
            print("❌ JavaScript file not accessible")
            
        return True
        
    except Exception as e:
        print(f"❌ Browser simulation failed: {str(e)}")
        return False

def main():
    """Run all upload tests"""
    print("AlgoJury Upload Diagnostic Test")
    print("=" * 50)
    
    # Create test files
    dataset_file, model_file = create_test_files()
    
    # Run tests
    tests = [
        ("Server Connection", test_server_connection),
        ("Browser Simulation", test_browser_simulation),
        ("Get Columns Endpoint", lambda: test_get_columns_endpoint(dataset_file)),
        ("Full Upload Process", lambda: test_full_upload(dataset_file, model_file))
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} crashed: {str(e)}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    all_passed = all(results.values())
    if all_passed:
        print("\n🎉 All tests passed! Upload functionality is working.")
        print("If users are still having issues, it might be:")
        print("   • Browser caching issues (try hard refresh: Ctrl+F5)")
        print("   • JavaScript disabled in browser")
        print("   • Browser compatibility issues")
        print("   • Network/firewall blocking requests")
    else:
        print("\n⚠️ Some tests failed. Issues detected in upload functionality.")
    
    # Cleanup
    for file in [dataset_file, model_file]:
        if os.path.exists(file):
            os.remove(file)
            print(f"Cleaned up: {file}")

if __name__ == "__main__":
    main()