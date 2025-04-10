import CustomFooter from 'components/Footer/Footer';
import CustomHeader from 'components/Header/Header';
import { ROUTE } from 'constants/routes';
import { useRouter } from 'expo-router';
import { useState } from 'react';
import { View, Dimensions, Text } from 'react-native';
import { Calendar, LocaleConfig } from 'react-native-calendars';

const { width } = Dimensions.get('window');

LocaleConfig.locales['ko'] = {
  monthNames: [
    '1월',
    '2월',
    '3월',
    '4월',
    '5월',
    '6월',
    '7월',
    '8월',
    '9월',
    '10월',
    '11월',
    '12월',
  ],
  monthNamesShort: [
    '1월',
    '2월',
    '3월',
    '4월',
    '5월',
    '6월',
    '7월',
    '8월',
    '9월',
    '10월',
    '11월',
    '12월',
  ],
  dayNames: ['일요일', '월요일', '화요일', '수요일', '목요일', '금요일', '토요일'],
  dayNamesShort: ['일', '월', '화', '수', '목', '금', '토'],
  today: '오늘',
};

LocaleConfig.defaultLocale = 'ko';

export default function NewsletterCalendar() {
  const toKSTString = (date: Date) => {
    const koreaTimeOffset = 9 * 60 * 60 * 1000;
    return new Date(date.getTime() + koreaTimeOffset).toISOString().split('T')[0];
  };
  const now = new Date();
  const todayString = toKSTString(now);
  const minDateString = '2025-04-01';
  const isBefore6PM = now.getHours() < 18;
  const maxDateString = isBefore6PM
    ? toKSTString(new Date(now.getTime() - 24 * 60 * 60 * 1000))
    : todayString;
  const [currentDate, setCurrentDate] = useState(todayString);
  const router = useRouter();

  const isSameMonth = (curDate: string, targetDate: string) => {
    return (
      new Date(curDate).getFullYear() === new Date(targetDate).getFullYear() &&
      new Date(curDate).getMonth() === new Date(targetDate).getMonth()
    );
  };

  const onPressDate = (date: any) => {
    const newDate = date.dateString.replaceAll('-', '').slice(2);
    router.navigate(`${ROUTE.NEWSLETTER.INDEX}/${newDate}`);
  };

  const onGoBack = () => {
    router.navigate(ROUTE.HOME);
  };

  return (
    <>
      <CustomHeader title="뉴스레터" onGoBack={onGoBack} />
      <View className="h-full w-full items-center gap-6 pt-24">
        <Text className="text-lg font-bold text-text">
          날짜를 클릭하면 뉴스레터를 볼 수 있어요!
        </Text>
        <Calendar
          onDayPress={onPressDate}
          style={{
            width: width * 0.9,
            borderWidth: 1,
            borderColor: '#E5E7EB',
            borderRadius: 12,
            padding: 10,
            backgroundColor: 'white',
          }}
          theme={{
            textDayFontSize: 16,
            textMonthFontSize: 18,
            textDayHeaderFontSize: 14,
            selectedDayBackgroundColor: '#3B82F6',
            selectedDayTextColor: 'white',
            arrowColor: '#3B82F6',
            monthTextColor: '#111827',
            textSectionTitleColor: '#6B7280',
          }}
          current={currentDate}
          minDate={minDateString}
          // maxDate={maxDateString}
          hideExtraDays
          onMonthChange={(date: any) => {
            const newDate = `${date.year}-${String(date.month).padStart(2, '0')}-01`;
            setCurrentDate(newDate);
          }}
          disableArrowLeft={isSameMonth(currentDate, minDateString)}
          disableArrowRight={isSameMonth(currentDate, maxDateString)}
        />
      </View>
      <CustomFooter />
    </>
  );
}
