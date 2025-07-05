def handler(event):
    """V136 KOREAN SAFE: Main handler - NEW WORKFLOW COMPATIBLE"""
    try:
        print("=== V136 Handler Started - NEW WORKFLOW ===")
        
        # Get input data
        input_data = event.get('input', event)
        
        # NEW: Check handler_type first
        handler_type = input_data.get('handler_type', 'detail')
        
        print(f"Handler type: {handler_type}")
        print(f"Input keys: {sorted(input_data.keys())}")
        
        # Route based on handler_type
        if handler_type == 'text_section':
            # NEW: Handle text sections from Claude
            route_number = input_data.get('route_number', 7)
            text_type = input_data.get('text_type', 'md_talk')
            
            # Download Korean font if needed
            if not os.path.exists('/tmp/NanumGothic.ttf'):
                print("Downloading Korean font for text section...")
                if not download_korean_font():
                    print("WARNING: Korean font download failed, using fallback")
            
            # Get text from various fields
            claude_text = (
                input_data.get('md_talk') or 
                input_data.get('design_point') or
                input_data.get('claude_text') or 
                input_data.get('text') or 
                ''
            )
            
            print(f"Processing text_section: route={route_number}, type={text_type}")
            print(f"Text content: {claude_text[:100]}...")
            
            # Create text section
            if route_number == 7 or 'md' in text_type.lower():
                detail_page = create_ai_generated_md_talk(claude_text)
                page_type = "md_talk"
                actual_route = 7
            else:
                detail_page = create_ai_generated_design_point(claude_text)
                page_type = "design_point"
                actual_route = 8
            
            # Convert to base64
            buffered = BytesIO()
            detail_page.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue())
            detail_base64_no_padding = img_str.decode('utf-8').rstrip('=')
            
            # Send to webhook
            file_name = f"{page_type}_route_{actual_route}.png"
            metadata = {
                "enhanced_image": detail_base64_no_padding,
                "status": "success",
                "handler_type": handler_type,
                "page_type": page_type,
                "route_number": actual_route,
                "text_type": text_type,
                "version": "V136_NEW_WORKFLOW"
            }
            
            webhook_result = send_to_webhook(
                detail_base64_no_padding, 
                handler_type, 
                file_name, 
                actual_route, 
                metadata
            )
            
            # Return
            return {
                "output": metadata
            }
            
        elif handler_type == 'detail':
            # Original detail page logic continues here
            print("Processing detail handler (original logic)")
            
            # Download Korean font
            if not os.path.exists('/tmp/NanumGothic.ttf'):
                print("Downloading Korean font...")
                if not download_korean_font():
                    print("WARNING: Korean font download failed, using fallback")
            else:
                print("Korean font already exists")
            
            print("=== Input Data ===")
            print(f"Keys: {sorted(input_data.keys())}")
            print(f"route_number: {input_data.get('route_number', 'None')}")
            
            # Detect group
            group_number = detect_group_number_from_input(input_data)
            
            print(f"\n=== Detection Result ===")
            print(f"Detected group: {group_number}")
            
            # Route number override
            route_str = str(input_data.get('route_number', '0'))
            try:
                route_int = int(route_str) if route_str.isdigit() else 0
            except:
                route_int = 0
                
            if route_int > 0:
                original_group = group_number
                group_number = route_int
                if original_group != group_number:
                    print(f"!!! OVERRIDE: Group {original_group} → {group_number}")
            
            print(f"\n=== FINAL GROUP: {group_number} ===")
            
            # Validate group number
            if group_number == 0:
                raise ValueError(f"Could not determine group number. Keys: {list(input_data.keys())}")
            
            if group_number < 1 or group_number > 8:
                raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
            
            # Process image input format
            if 'image' in input_data and input_data['image']:
                print(f"Found 'image' key: {str(input_data['image'])[:100]}...")
                image_data = input_data['image']
                
                if ';' in str(image_data):
                    urls = str(image_data).split(';')
                    input_data['images'] = []
                    for url in urls:
                        url = url.strip()
                        if url:
                            input_data['images'].append({'url': url})
                    print(f"Converted 'image' to {len(input_data['images'])} image array")
                else:
                    input_data['url'] = image_data
                    print("Set single URL from 'image' key")
            
            if 'combined_urls' in input_data and input_data['combined_urls']:
                urls = str(input_data['combined_urls']).split(';')
                input_data['images'] = []
                for url in urls:
                    url = url.strip()
                    if url:
                        input_data['images'].append({'url': url})
                print(f"Converted combined_urls to {len(input_data['images'])} images")
            
            # Handle specific image keys
            if not input_data.get('images') and not input_data.get('url'):
                if group_number == 6:
                    # Group 6 special handling
                    for key in ['image9', 'image6', 'group6', 'color_image']:
                        if key in input_data and input_data[key]:
                            input_data['url'] = input_data[key]
                            print(f"V136 Group 6: Using {key}")
                            break
                
                elif f'image{group_number}' in input_data:
                    image_url = input_data[f'image{group_number}']
                    if ';' in str(image_url):
                        urls = str(image_url).split(';')
                        input_data['images'] = [{'url': url.strip()} for url in urls if url.strip()]
                    else:
                        input_data['url'] = image_url
                    print(f"Processed image{group_number}")
                
                elif group_number == 3:
                    images_to_add = []
                    if 'image3' in input_data:
                        images_to_add.append({'url': input_data['image3']})
                    if 'image4' in input_data:
                        images_to_add.append({'url': input_data['image4']})
                    if images_to_add:
                        input_data['images'] = images_to_add
                        print(f"Group 3: Found {len(images_to_add)} images")
                
                elif group_number == 4:
                    images_to_add = []
                    if 'image5' in input_data:
                        images_to_add.append({'url': input_data['image5']})
                    if 'image6' in input_data and 'image5' in input_data:
                        images_to_add.append({'url': input_data['image6']})
                    if images_to_add:
                        input_data['images'] = images_to_add
                        print(f"Group 4: Found {len(images_to_add)} images")
                
                elif group_number == 5:
                    images_to_add = []
                    if 'image7' in input_data:
                        images_to_add.append({'url': input_data['image7']})
                    if 'image8' in input_data:
                        images_to_add.append({'url': input_data['image8']})
                    if images_to_add:
                        input_data['images'] = images_to_add
                        print(f"Group 5: Found {len(images_to_add)} images")
            
            # Process by group
            if group_number == 6:
                print("V136: Processing Group 6 COLOR section")
                detail_page = process_color_section(input_data)
                page_type = "color_section"
                
            elif group_number == 7:
                print("V136: Processing Group 7 MD TALK text section")
                detail_page, section_type = process_korean_text_section(input_data, 7)
                page_type = f"text_section_{section_type}"
                
            elif group_number == 8:
                print("V136: Processing Group 8 DESIGN POINT text section")
                detail_page, section_type = process_korean_text_section(input_data, 8)
                page_type = f"text_section_{section_type}"
                
            elif group_number in [1, 2]:
                print(f"V136: Processing Group {group_number} single image")
                detail_page = process_single_image(input_data, group_number)
                page_type = "individual"
                
            elif group_number in [3, 4, 5]:
                print(f"V136: Processing Group {group_number} combined images")
                if 'images' not in input_data or not isinstance(input_data['images'], list):
                    input_data['images'] = [input_data]
                
                if group_number == 5:
                    detail_page = process_clean_combined_images(input_data.get('images', []), group_number, input_data)
                else:
                    detail_page = process_clean_combined_images(input_data['images'], group_number, input_data)
                
                page_type = "clean_combined"
            
            else:
                raise ValueError(f"Invalid group number: {group_number}. Must be 1-8.")
            
            # Convert to base64
            buffered = BytesIO()
            detail_page.save(buffered, format="PNG", optimize=True)
            img_str = base64.b64encode(buffered.getvalue())
            
            detail_base64 = img_str.decode('utf-8')
            detail_base64_no_padding = detail_base64.rstrip('=')
            
            print(f"V136: Detail page created: {detail_page.size}")
            print(f"V136: Base64 length: {len(detail_base64_no_padding)} chars")
            
            # Metadata
            metadata = {
                "enhanced_image": detail_base64_no_padding,
                "status": "success",
                "page_type": korean_safe_string_encode(page_type),
                "page_number": group_number,
                "route_number": group_number,
                "actual_group": group_number,
                "dimensions": {
                    "width": detail_page.width,
                    "height": detail_page.height
                },
                "version": "V136_KOREAN_SAFE_FIXED",
                "image_count": len(input_data.get('images', [input_data])),
                "processing_time": "calculated_later",
                "font_status": "korean_font_available" if os.path.exists('/tmp/NanumGothic.ttf') else "fallback_font",
                "korean_support": "100_percent_safe",
                "group_info": {
                    "requested": group_number,
                    "detected": detect_group_number_from_input(input_data),
                    "final": group_number
                }
            }
            
            # Send to webhook
            file_name = f"detail_group_{group_number}.png"
            webhook_result = send_to_webhook(detail_base64_no_padding, "detail", file_name, group_number, metadata)
            
            # Return response
            return {
                "output": metadata
            }
            
        else:
            raise ValueError(f"Unknown handler_type: {handler_type}")
            
    except Exception as e:
        # Error handling
        error_msg = korean_safe_string_encode(f"Handler failed: {str(e)}")
        print(f"V136 ERROR: {error_msg}")
        traceback_str = korean_safe_string_encode(traceback.format_exc())
        print(f"V136 TRACEBACK: {traceback_str}")
        
        return {
            "output": {
                "error": error_msg,
                "status": "error",
                "traceback": traceback_str,
                "version": "V136_NEW_WORKFLOW",
                "handler_type": handler_type if 'handler_type' in locals() else "unknown",
                "input_keys": list(event.get('input', event).keys()) if isinstance(event.get('input', event), dict) else []
            }
        }
